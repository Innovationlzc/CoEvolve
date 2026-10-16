"""Stable, dependency-free contracts for comparative experiment runtimes.

The experiment package deliberately does not import the production controller.
The real deployment and deterministic tests both implement these small ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Mapping

from ..provider_routes import get_provider_route


_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ProtocolBinding:
    """Pure-data seam to frozen code plus every externally configurable API.

    It records identity only.  Verifying a live Git tree belongs to the outer
    deployment launcher, not this independent engine.  The extra fingerprints
    prevent callbacks, instructions, prompts, or SONIC lowering from drifting
    outside the pinned Git tree while still claiming a formal run.
    """

    protocol_id: str = "unbound_smoke_protocol"
    base_method_commit: str = "unbound_smoke_commit"
    invariant_fingerprint: str = "unbound"
    terminal_fingerprint: str = "unbound"
    provider_route_id: str = "unbound"
    provider_route_fingerprint: str = "unbound"
    observation_contract_fingerprint: str = "unbound"
    task_instruction_fingerprint: str = "unbound"
    shared_prompt_contract_fingerprint: str = "unbound"
    decision_prompt_fingerprint: str = "unbound"
    skill_api_fingerprint: str = "unbound"

    @property
    def formally_pinned(self) -> bool:
        try:
            route_fingerprint = get_provider_route(
                self.provider_route_id
            ).fingerprint
        except ValueError:
            route_fingerprint = ""
        return bool(
            self.protocol_id
            and not self.protocol_id.startswith("unbound")
            and _GIT_COMMIT_RE.fullmatch(self.base_method_commit)
            and _SHA256_RE.fullmatch(self.invariant_fingerprint)
            and _SHA256_RE.fullmatch(self.terminal_fingerprint)
            and self.provider_route_fingerprint == route_fingerprint
            and _SHA256_RE.fullmatch(self.observation_contract_fingerprint)
            and _SHA256_RE.fullmatch(self.task_instruction_fingerprint)
            and _SHA256_RE.fullmatch(self.shared_prompt_contract_fingerprint)
            and _SHA256_RE.fullmatch(self.decision_prompt_fingerprint)
            and _SHA256_RE.fullmatch(self.skill_api_fingerprint)
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """One host-owned observation at a physical transaction boundary."""

    sequence: int
    state: Mapping[str, Any] = field(default_factory=dict)
    images: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise TypeError("Observation.sequence must be a non-negative int")
        _require_mapping("Observation.state", self.state)
        _require_mapping("Observation.images", self.images)
        _require_mapping("Observation.metadata", self.metadata)


@dataclass(frozen=True, slots=True)
class RobotIntent:
    """One robot's part of an atomic cooperative transaction."""

    actor: str
    primitive: str
    objective: str
    mode: str = "default"
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_mapping("RobotIntent.arguments", self.arguments)


@dataclass(frozen=True, slots=True)
class TeamIntent:
    """Method-independent semantic contract presented to the host validator."""

    commands: tuple[RobotIntent, ...]
    atomic: bool = True
    lifecycle_phase: str = "unspecified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_strict_bool("TeamIntent.atomic", self.atomic)
        _require_mapping("TeamIntent.metadata", self.metadata)


@dataclass(frozen=True, slots=True)
class PolicyProposal:
    """Executable source plus a redundant, machine-checkable intent."""

    source: str
    intent: TeamIntent
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidatedPolicy:
    """A proposal accepted by the static policy/intent gate."""

    proposal: PolicyProposal
    canonical_ast: str
    ast_fingerprint: str


@dataclass(frozen=True, slots=True)
class PhysicalDispatchResult:
    """What happened at the frozen motor-backend boundary."""

    physically_dispatched: bool
    physical_transaction_completed: bool
    observation: Observation | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        _require_strict_bool(
            "PhysicalDispatchResult.physically_dispatched",
            self.physically_dispatched,
        )
        _require_strict_bool(
            "PhysicalDispatchResult.physical_transaction_completed",
            self.physical_transaction_completed,
        )
        if self.physical_transaction_completed and not self.physically_dispatched:
            raise ValueError(
                "a completed physical transaction must have been dispatched"
            )
        if self.observation is not None and not isinstance(
            self.observation, Observation
        ):
            raise TypeError("PhysicalDispatchResult.observation must be Observation")
        _require_mapping("PhysicalDispatchResult.evidence", self.evidence)
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("PhysicalDispatchResult.error must be a string")

    @property
    def dispatched(self) -> bool:
        return self.physically_dispatched

    @property
    def completed(self) -> bool:
        return self.physical_transaction_completed


@dataclass(frozen=True, slots=True)
class VerifierOutcome:
    """Physics-first acceptance and terminal evidence for one dispatch."""

    accepted: bool
    terminal: bool = False
    failure_code: str | None = None
    report: str = ""
    phase_completion: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_strict_bool("VerifierOutcome.accepted", self.accepted)
        _require_strict_bool("VerifierOutcome.terminal", self.terminal)
        if self.terminal and not self.accepted:
            raise ValueError("a terminal verifier outcome must be accepted")
        _require_mapping("VerifierOutcome.evidence", self.evidence)


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    """Opaque checkpoint identity owned by the deployment callback."""

    checkpoint_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string")
        _require_mapping("CheckpointRef.metadata", self.metadata)

    def __str__(self) -> str:
        return self.checkpoint_id


@dataclass(frozen=True, slots=True)
class RollbackResult:
    """Visible result of restore, planner reset, and recovery verification."""

    restored: bool
    recovery_verified: bool
    observation: Observation | None = None
    report: str = ""

    def __post_init__(self) -> None:
        _require_strict_bool("RollbackResult.restored", self.restored)
        _require_strict_bool(
            "RollbackResult.recovery_verified", self.recovery_verified
        )
        if self.recovery_verified and not self.restored:
            raise ValueError("recovery cannot be verified when restore failed")
        if self.observation is not None and not isinstance(
            self.observation, Observation
        ):
            raise TypeError("RollbackResult.observation must be Observation")


class FeedbackKind(str, Enum):
    VERIFIED_ACCEPTED = "verified_accepted"
    VERIFIED_REJECTED = "verified_rejected"
    INVALID_POLICY = "invalid_policy"
    EXACT_AST_BLOCKED = "exact_ast_blocked"
    PREDISPATCH_FAILURE = "predispatch_failure"
    INCOMPLETE_DISPATCH = "incomplete_dispatch"
    GENERATION_RECOVERY = "generation_recovery"


@dataclass(frozen=True, slots=True)
class CandidateFeedback:
    """Uniform feedback consumed by every method adapter."""

    kind: FeedbackKind
    report: str
    proposal: PolicyProposal | None = None
    ast_fingerprint: str | None = None
    verifier: VerifierOutcome | None = None

    @property
    def accepted(self) -> bool:
        return self.kind is FeedbackKind.VERIFIED_ACCEPTED


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    """One physically dispatched rollout and all host-owned evidence."""

    transaction_index: int
    checkpoint: CheckpointRef
    observation_before: Observation
    policy: ValidatedPolicy
    dispatch: PhysicalDispatchResult
    verifier: VerifierOutcome
    observation_after: Observation | None


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """Latest physical failure retained independently of dynamic memory."""

    transaction_index: int
    checkpoint_id: str
    ast_fingerprint: str
    failure_code: str
    report: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AcceptedManifestEntry:
    """Contiguous verifier-accepted checkpoint lineage."""

    transaction_index: int
    parent_checkpoint_id: str
    checkpoint_id: str
    ast_fingerprint: str
    intent: TeamIntent
    verifier: VerifierOutcome


@dataclass(frozen=True, slots=True)
class AuditEvent:
    sequence: int
    kind: str
    checkpoint_id: str | None
    transaction_index: int | None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunAccounting:
    """Mutable counters copied into the immutable final result."""

    physical_transactions: int = 0
    completed_physical_transactions: int = 0
    accepted_transactions: int = 0
    rejected_transactions: int = 0
    incomplete_transactions: int = 0
    predispatch_failures: int = 0
    invalid_candidates: int = 0
    exact_blocked_candidates: int = 0
    checkpoints_saved: int = 0
    rollbacks: int = 0

    def snapshot(self) -> "RunAccounting":
        return RunAccounting(
            physical_transactions=self.physical_transactions,
            completed_physical_transactions=self.completed_physical_transactions,
            accepted_transactions=self.accepted_transactions,
            rejected_transactions=self.rejected_transactions,
            incomplete_transactions=self.incomplete_transactions,
            predispatch_failures=self.predispatch_failures,
            invalid_candidates=self.invalid_candidates,
            exact_blocked_candidates=self.exact_blocked_candidates,
            checkpoints_saved=self.checkpoints_saved,
            rollbacks=self.rollbacks,
        )


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ADAPTER_EXHAUSTED = "adapter_exhausted"
    ROLLBACK_FAILED = "rollback_failed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    terminal_reason: str
    observation: Observation
    checkpoint: CheckpointRef
    accepted_manifest: tuple[AcceptedManifestEntry, ...]
    raw_audit: tuple[AuditEvent, ...]
    latest_failure: FailureRecord | None
    accounting: RunAccounting
    usage: Mapping[str, Any]

    @property
    def success(self) -> bool:
        return self.status is RunStatus.SUCCEEDED


Observer = Callable[[], Observation]
Dispatcher = Callable[[ValidatedPolicy], PhysicalDispatchResult]
Verifier = Callable[
    [Observation, ValidatedPolicy, PhysicalDispatchResult], VerifierOutcome
]
CheckpointSaver = Callable[[Observation], CheckpointRef | str]
Rollback = Callable[[CheckpointRef, Observation], RollbackResult | Observation | bool]
TerminalPredicate = Callable[[Observation, VerifierOutcome], bool]
AuditSink = Callable[[AuditEvent], None]


@dataclass(frozen=True, slots=True)
class RuntimeCallbacks:
    """Ports implemented by either the real deployment or a scripted fake."""

    observe: Observer
    dispatch: Dispatcher
    verify: Verifier
    save_checkpoint: CheckpointSaver
    rollback: Rollback
    terminal_predicate: TerminalPredicate | None = None
    audit_sink: AuditSink | None = None


@dataclass(frozen=True, slots=True)
class EngineConfig:
    physical_budget: int = 80
    max_candidate_attempts: int | None = None
    max_generation_recovery_cycles: int = 3
    protocol: ProtocolBinding = ProtocolBinding()
    require_formal_protocol_pin: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.physical_budget, bool)
            or not isinstance(self.physical_budget, int)
            or self.physical_budget <= 0
        ):
            raise ValueError("physical_budget must be a positive integer")
        if (
            isinstance(self.max_generation_recovery_cycles, bool)
            or not isinstance(self.max_generation_recovery_cycles, int)
            or self.max_generation_recovery_cycles <= 0
        ):
            raise ValueError(
                "max_generation_recovery_cycles must be a positive integer"
            )
        if self.max_candidate_attempts is not None and (
            isinstance(self.max_candidate_attempts, bool)
            or not isinstance(self.max_candidate_attempts, int)
            or self.max_candidate_attempts <= 0
        ):
            raise ValueError(
                "max_candidate_attempts must be a positive integer when set"
            )
        if self.require_formal_protocol_pin and not self.protocol.formally_pinned:
            raise ValueError(
                "formal run requires a pinned protocol, base commit, and live contracts"
            )
        if self.require_formal_protocol_pin and self.physical_budget != 80:
            raise ValueError("the frozen formal physical budget must be 80")
        if (
            self.require_formal_protocol_pin
            and self.max_candidate_attempts is not None
        ):
            raise ValueError(
                "formal max_candidate_attempts is fixed by the common runtime"
            )


def _require_strict_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _require_mapping(name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
