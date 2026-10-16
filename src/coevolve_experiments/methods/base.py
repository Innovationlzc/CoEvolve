"""Method adapter boundary shared by all Table-2 conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.live import ExecutionCapability, PolicyRepresentation
from coevolve_experiments.runtime.contracts import (
    AcceptedManifestEntry,
    CandidateFeedback,
    CheckpointRef,
    FailureRecord,
    Observation,
    PolicyProposal,
    RolloutRecord,
    RunAccounting,
)


class MethodRole(str, Enum):
    TASK_ORCHESTRATOR = "task_orchestrator"
    POLICY_GENERATOR = "policy_generator"
    EVOLUTION_MANAGER = "evolution_manager"
    MUTATION_OPERATOR = "mutation_operator"
    MEMORY_UPDATE = "memory_update"

    FLAT_TASK_PLANNER = "flat_task_planner"
    FLAT_STRATEGY_SELECTOR = "flat_strategy_selector"
    FLAT_POLICY_GENERATOR = "flat_policy_generator"
    REFLECTIVE_REWRITE = "reflective_rewrite"

    REMAC_PLANNER = "remac_planner"
    REMAC_PRECONDITION_REFLECTION = "remac_precondition_reflection"
    REMAC_POSTCONDITION_REFLECTION = "remac_postcondition_reflection"
    REMAC_REPLANNER = "remac_replanner"

    COHERENT_CENTRAL_PROPOSER = "coherent_central_proposer"
    COHERENT_FRONT_FEASIBILITY = "coherent_front_feasibility"
    COHERENT_BACK_FEASIBILITY = "coherent_back_feasibility"
    COHERENT_FEEDBACK_ADJUSTER = "coherent_feedback_adjuster"


@dataclass(frozen=True, slots=True)
class RoleUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class RoleResponse:
    content: Any
    usage: RoleUsage = RoleUsage()
    raw: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RoleBackend(Protocol):
    """Only model seam used by adapters; roles are explicit and auditable."""

    def invoke(self, role: str, payload: Mapping[str, Any]) -> RoleResponse:
        ...


@dataclass(frozen=True, slots=True)
class MemorySwitches:
    strategy_retrieval: bool
    strategy_promotion: bool
    action_exemplar_retrieval: bool
    action_exemplar_promotion: bool

    @classmethod
    def enabled(cls) -> "MemorySwitches":
        return cls(True, True, True, True)

    @classmethod
    def disabled(cls) -> "MemorySwitches":
        return cls(False, False, False, False)

    @property
    def all_disabled(self) -> bool:
        return not any(
            (
                self.strategy_retrieval,
                self.strategy_promotion,
                self.action_exemplar_retrieval,
                self.action_exemplar_promotion,
            )
        )


@dataclass(frozen=True, slots=True)
class SafetyInvariants:
    physics_verifier: bool = True
    checkpoint: bool = True
    rollback: bool = True
    raw_rollout: bool = True
    accepted_manifest: bool = True
    latest_physical_failure: bool = True
    checkpoint_local_exact_ast_blocker: bool = True

    @property
    def all_enabled(self) -> bool:
        return all(
            (
                self.physics_verifier,
                self.checkpoint,
                self.rollback,
                self.raw_rollout,
                self.accepted_manifest,
                self.latest_physical_failure,
                self.checkpoint_local_exact_ast_blocker,
            )
        )


REQUIRED_SAFETY_INVARIANTS = SafetyInvariants()


@dataclass(frozen=True, slots=True)
class MethodDescriptor:
    variant: MethodVariant
    allowed_roles: frozenset[MethodRole]
    hierarchical_evolution: bool
    durable_dynamic_memory: bool
    memory_switches: MemorySwitches
    safety_invariants: SafetyInvariants = REQUIRED_SAFETY_INVARIANTS
    policy_representation: PolicyRepresentation = PolicyRepresentation.TEAM_INTENT_IR
    execution_capability: ExecutionCapability = ExecutionCapability.CONTRACT_SMOKE_ONLY


@dataclass(frozen=True, slots=True)
class AdapterContext:
    instruction: str
    observation: Observation
    checkpoint: CheckpointRef
    remaining_physical_budget: int
    accepted_manifest: tuple[AcceptedManifestEntry, ...]
    latest_physical_failure: FailureRecord | None
    blocked_ast_fingerprints: frozenset[str]
    accounting: RunAccounting


@dataclass(frozen=True, slots=True)
class ActionExemplarQuery:
    """Host-owned lookup key created only after a strategy is committed.

    ``primitive``, ``semantic_objective``, and ``coordination_mode`` are exact
    compatibility keys.  ``active_phase`` is deliberately a ranking hint, not
    an alias or compatibility key.
    """

    primitive: str
    semantic_objective: str
    coordination_mode: str
    active_phase: str

    def __post_init__(self) -> None:
        for name in (
            "primitive",
            "semantic_objective",
            "coordination_mode",
            "active_phase",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ActionExemplarQuery.{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """Host-visible separation between plan state and the action-time strategy.

    The task plan persists across transactions.  The committed strategy is the
    *single* coordination decision that the Coder must ground for the current
    transaction.  Keeping the two values separate prevents a model-authored
    wrapper (or a stale strategy nested in a previous plan) from becoming the
    lookup key for executable action exemplars.
    """

    updated_plan: Any
    committed_strategy: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AdapterUsageSummary:
    model_calls: int
    input_tokens: int
    output_tokens: int
    calls_by_role: Mapping[str, int]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "calls_by_role": dict(self.calls_by_role),
        }


@runtime_checkable
class AdapterMemoryPort(Protocol):
    """Optional strategy/action store; safety state never passes this port."""

    def retrieve_strategies(
        self, context: AdapterContext
    ) -> Sequence[Mapping[str, Any]]:
        ...

    def retrieve_action_exemplars(
        self,
        context: AdapterContext,
        query: ActionExemplarQuery,
    ) -> Sequence[Mapping[str, Any]]:
        ...

    def promote_strategy(
        self, proposal: Any, rollout: RolloutRecord
    ) -> None:
        ...

    def promote_action_exemplar(
        self, proposal: Any, rollout: RolloutRecord
    ) -> None:
        ...



@runtime_checkable
class AdapterMemoryLifecycle(Protocol):
    """Optional lifecycle extension implemented by audited file stores."""

    def finalize(self) -> Mapping[str, Any] | None:
        ...


class NullAdapterMemory:
    """Default no-op implementation, useful before a store is wired in."""

    def retrieve_strategies(
        self, context: AdapterContext
    ) -> Sequence[Mapping[str, Any]]:
        return ()

    def retrieve_action_exemplars(
        self,
        context: AdapterContext,
        query: ActionExemplarQuery,
    ) -> Sequence[Mapping[str, Any]]:
        del context, query
        return ()

    def promote_strategy(self, proposal: Any, rollout: RolloutRecord) -> None:
        return None

    def promote_action_exemplar(
        self, proposal: Any, rollout: RolloutRecord
    ) -> None:
        return None

    def finalize(self) -> Mapping[str, Any] | None:
        return None


@runtime_checkable
class MethodAdapter(Protocol):
    descriptor: MethodDescriptor

    def propose_initial(self, context: AdapterContext) -> PolicyProposal | None:
        ...

    def propose_next(
        self, context: AdapterContext, feedback: CandidateFeedback
    ) -> PolicyProposal | None:
        ...

    def on_verified_rollout(
        self, rollout: RolloutRecord, context: AdapterContext
    ) -> Mapping[str, Any] | None:
        ...

    def usage_summary(self) -> AdapterUsageSummary:
        ...

    def finalize_memory(self) -> Mapping[str, Any] | None:
        ...


class AdapterContractError(RuntimeError):
    pass


class BaseMethodAdapter:
    """Role enforcement, usage accounting, and memory-update plumbing."""

    descriptor: MethodDescriptor

    def __init__(
        self,
        backend: RoleBackend,
        *,
        memory: AdapterMemoryPort | None = None,
    ) -> None:
        self._backend = backend
        self._memory = memory if memory is not None else NullAdapterMemory()
        self._calls_by_role: dict[str, int] = {}
        self._input_tokens = 0
        self._output_tokens = 0
        self._memory_update_audit: list[Mapping[str, Any]] = []

    def _invoke(self, role: MethodRole, payload: Mapping[str, Any]) -> RoleResponse:
        if role not in self.descriptor.allowed_roles:
            raise AdapterContractError(
                f"{self.descriptor.variant.value} cannot invoke role {role.value}"
            )
        response = self._backend.invoke(role.value, dict(payload))
        if not isinstance(response, RoleResponse):
            raise AdapterContractError("RoleBackend.invoke must return RoleResponse")
        self._calls_by_role[role.value] = self._calls_by_role.get(role.value, 0) + 1
        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens
        return response

    def _begin_run(self) -> None:
        """Reset adapter-owned run state without touching an injected bank."""

        self._calls_by_role.clear()
        self._input_tokens = 0
        self._output_tokens = 0
        self._memory_update_audit.clear()

    def _strategy_memory_context(self, context: AdapterContext) -> dict[str, Any]:
        strategies: Sequence[Mapping[str, Any]] = ()
        if self.descriptor.memory_switches.strategy_retrieval:
            strategies = self._memory.retrieve_strategies(context)
        return {"strategy_records": list(strategies)}

    def _action_memory_context(
        self,
        context: AdapterContext,
        committed_strategy: object,
    ) -> dict[str, Any]:
        query = action_exemplar_query(committed_strategy, context)
        exemplars: Sequence[Mapping[str, Any]] = ()
        if self.descriptor.memory_switches.action_exemplar_retrieval:
            exemplars = self._memory.retrieve_action_exemplars(context, query)
        return {
            "committed_strategy": _committed_strategy_mapping(
                committed_strategy
            ),
            "action_exemplar_query": {
                "primitive": query.primitive,
                "semantic_objective": query.semantic_objective,
                "coordination_mode": query.coordination_mode,
                "active_phase": query.active_phase,
            },
            "action_exemplars": list(exemplars),
        }

    def _common_payload(self, context: AdapterContext) -> dict[str, Any]:
        latest = context.latest_physical_failure
        return {
            "instruction": context.instruction,
            "observation": dict(context.observation.state),
            "images": dict(context.observation.images),
            "observation_metadata": dict(context.observation.metadata),
            "observation_sequence": context.observation.sequence,
            "checkpoint_id": context.checkpoint.checkpoint_id,
            "remaining_physical_budget": context.remaining_physical_budget,
            "accepted_manifest_length": len(context.accepted_manifest),
            "latest_physical_failure": None
            if latest is None
            else {
                "failure_code": latest.failure_code,
                "report": latest.report,
                "ast_fingerprint": latest.ast_fingerprint,
            },
            "blocked_ast_fingerprints": sorted(context.blocked_ast_fingerprints),
        }

    def _orchestration_payload(self, context: AdapterContext) -> dict[str, Any]:
        """Payload for strategy commitment/evolution; contains no code examples."""

        return {
            **self._common_payload(context),
            **self._strategy_memory_context(context),
        }

    def _policy_payload(
        self,
        context: AdapterContext,
        committed_strategy: object,
    ) -> dict[str, Any]:
        """Payload for the Coder after strategy commitment."""

        return {
            **self._common_payload(context),
            **self._action_memory_context(context, committed_strategy),
        }

    def _run_memory_update(
        self, rollout: RolloutRecord, context: AdapterContext
    ) -> Mapping[str, Any]:
        """Always solicit a proposal; switches gate retrieval/promotion only."""

        response = self._invoke(
            MethodRole.MEMORY_UPDATE,
            {
                **self._common_payload(context),
                "rollout": _rollout_payload(rollout),
                "promotion_enabled": {
                    "strategy": self.descriptor.memory_switches.strategy_promotion,
                    "action_exemplar": self.descriptor.memory_switches.action_exemplar_promotion,
                },
            },
        )
        proposal = response.content
        entry = {
            "role": MethodRole.MEMORY_UPDATE.value,
            "proposal": proposal,
            "strategy_promoted": False,
            "action_exemplar_promoted": False,
        }
        if (
            rollout.dispatch.physical_transaction_completed
            and self.descriptor.memory_switches.strategy_promotion
        ):
            self._memory.promote_strategy(proposal, rollout)
            entry["strategy_promoted"] = True
        if (
            rollout.dispatch.physical_transaction_completed
            and rollout.verifier.accepted
            and self.descriptor.memory_switches.action_exemplar_promotion
            and _action_exemplar_eligible(rollout)
        ):
            self._memory.promote_action_exemplar(proposal, rollout)
            entry["action_exemplar_promoted"] = True
        self._memory_update_audit.append(entry)
        return entry

    @property
    def memory_update_audit(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._memory_update_audit)

    def usage_summary(self) -> AdapterUsageSummary:
        return AdapterUsageSummary(
            model_calls=sum(self._calls_by_role.values()),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            calls_by_role=dict(self._calls_by_role),
        )

    def finalize_memory(self) -> Mapping[str, Any] | None:
        """Finalize an injected lifecycle store; no-op memories return ``None``."""

        finalize = getattr(self._memory, "finalize", None)
        if finalize is None or not callable(finalize):
            return None
        result = finalize()
        if result is not None and not isinstance(result, Mapping):
            raise AdapterContractError(
                "AdapterMemoryPort.finalize must return a mapping or None"
            )
        return result


def action_exemplar_query(
    committed_strategy: object,
    context: AdapterContext,
) -> ActionExemplarQuery:
    """Validate the retrieval fields of a committed coordination strategy."""

    strategy = _committed_strategy_mapping(committed_strategy)
    phase_value = context.observation.state.get("phase", "unspecified")
    phase = str(phase_value).strip() if phase_value is not None else ""
    if not phase:
        phase = "unspecified"
    return ActionExemplarQuery(
        primitive=_required_strategy_text(strategy, "primitive"),
        semantic_objective=_required_strategy_text(
            strategy,
            "semantic_objective",
        ),
        coordination_mode=_required_strategy_text(
            strategy,
            "coordination_mode",
        ),
        active_phase=phase,
    )


def orchestration_result_from_response(response: RoleResponse) -> OrchestrationResult:
    """Parse the exact output contract of an orchestration/mutation role.

    Legacy responses that expose only ``plan`` or bury a strategy inside an
    opaque plan are rejected.  Such a response cannot prove which strategy was
    committed before action-exemplar retrieval, so accepting it would make the
    memory condition ambiguous.
    """

    content = response.content
    if not isinstance(content, Mapping):
        raise AdapterContractError(
            "orchestration role must return an updated_plan and committed_strategy"
        )
    if "updated_plan" not in content:
        raise AdapterContractError("orchestration response requires updated_plan")
    updated_plan = content["updated_plan"]
    if updated_plan is None:
        raise AdapterContractError("orchestration updated_plan cannot be null")
    if "committed_strategy" not in content:
        raise AdapterContractError(
            "orchestration response requires committed_strategy"
        )
    strategy = _committed_strategy_mapping(content["committed_strategy"])
    return OrchestrationResult(
        updated_plan=updated_plan,
        committed_strategy=strategy,
    )


def flat_strategy_commit_from_response(
    response: RoleResponse,
) -> Mapping[str, Any]:
    """Parse a Flat selector response without allowing task-plan mutation.

    The Flat condition has a dedicated strategy-selection role on every
    transaction.  Its output may be either the strategy object itself or an
    otherwise-empty ``committed_strategy`` wrapper.  Plan-shaped fields are
    rejected at both levels so this role cannot quietly become a task-level
    replanner.
    """

    content = response.content
    if not isinstance(content, Mapping):
        raise AdapterContractError(
            "flat strategy selector must return a strategy mapping"
        )
    wrapper = dict(content)
    forbidden_plan_fields = frozenset(
        {
            "updated_plan",
            "task_plan",
            "fixed_task_plan",
            "plan",
            "phases",
        }
    )
    present = forbidden_plan_fields.intersection(wrapper)
    if present:
        raise AdapterContractError(
            "flat strategy selector cannot return task-plan fields: "
            + ", ".join(sorted(present))
        )
    if "committed_strategy" in wrapper and set(wrapper) != {
        "committed_strategy"
    }:
        raise AdapterContractError(
            "flat strategy wrapper may contain only committed_strategy"
        )
    strategy = _committed_strategy_mapping(wrapper)
    nested_plan_fields = forbidden_plan_fields.intersection(strategy)
    if nested_plan_fields:
        raise AdapterContractError(
            "flat committed strategy cannot contain task-plan fields: "
            + ", ".join(sorted(nested_plan_fields))
        )
    return strategy


def _committed_strategy_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterContractError(
            "strategy-producing role must return a committed strategy mapping"
        )
    nested = value.get("committed_strategy")
    strategy = nested if nested is not None else value
    if not isinstance(strategy, Mapping):
        raise AdapterContractError("committed_strategy must be a mapping")
    # Copy the complete strategy (roles/parameters included when present), but
    # validate the exact lookup triple before any action memory can be queried.
    for field_name in (
        "primitive",
        "semantic_objective",
        "coordination_mode",
    ):
        _required_strategy_text(strategy, field_name)
    return dict(strategy)


def _required_strategy_text(strategy: Mapping[str, Any], field_name: str) -> str:
    value = strategy.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise AdapterContractError(
            f"committed strategy requires non-empty {field_name}"
        )
    return value.strip()


def feedback_payload(feedback: CandidateFeedback) -> dict[str, Any]:
    verifier = feedback.verifier
    return {
        "kind": feedback.kind.value,
        "report": feedback.report,
        "ast_fingerprint": feedback.ast_fingerprint,
        "verifier": None
        if verifier is None
        else {
            "accepted": verifier.accepted,
            "terminal": verifier.terminal,
            "failure_code": verifier.failure_code,
            "report": verifier.report,
            "phase_completion": verifier.phase_completion,
            "evidence": dict(verifier.evidence),
        },
    }


def _rollout_payload(rollout: RolloutRecord) -> dict[str, Any]:
    return {
        "transaction_index": rollout.transaction_index,
        "checkpoint_id": rollout.checkpoint.checkpoint_id,
        "ast_fingerprint": rollout.policy.ast_fingerprint,
        "physically_dispatched": rollout.dispatch.physically_dispatched,
        "physical_transaction_completed": rollout.dispatch.physical_transaction_completed,
        "accepted": rollout.verifier.accepted,
        "terminal": rollout.verifier.terminal,
        "failure_code": rollout.verifier.failure_code,
        "report": rollout.verifier.report,
        "phase_completion": rollout.verifier.phase_completion,
    }


def _action_exemplar_eligible(rollout: RolloutRecord) -> bool:
    """Only measured route progress or terminal success can seed executable memory."""

    evidence = rollout.verifier.evidence
    return bool(
        rollout.verifier.terminal
        or evidence.get("route_progress") is True
        or evidence.get("action_exemplar_eligible") is True
    )
