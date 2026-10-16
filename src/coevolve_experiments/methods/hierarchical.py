"""Contract models of Full and no-dynamic decision structures (smoke only)."""

from __future__ import annotations

from typing import Any, Mapping

from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.runtime.contracts import (
    CandidateFeedback,
    PolicyProposal,
    RolloutRecord,
)

from .base import (
    AdapterContext,
    BaseMethodAdapter,
    MemorySwitches,
    MethodDescriptor,
    MethodRole,
    feedback_payload,
    orchestration_result_from_response,
)
from .parsing import proposal_from_response


_FULL_HIERARCHICAL_ROLES = frozenset(
    {
        MethodRole.TASK_ORCHESTRATOR,
        MethodRole.POLICY_GENERATOR,
        MethodRole.EVOLUTION_MANAGER,
        MethodRole.MUTATION_OPERATOR,
        MethodRole.MEMORY_UPDATE,
    }
)

_NO_DYNAMIC_HIERARCHICAL_ROLES = frozenset(
    role for role in _FULL_HIERARCHICAL_ROLES if role is not MethodRole.MEMORY_UPDATE
)


class FullCoEvolveAdapter(BaseMethodAdapter):
    """Persistent task plan plus verifier-selected mutation granularity."""

    descriptor = MethodDescriptor(
        variant=MethodVariant.COEVOLE_FULL,
        allowed_roles=_FULL_HIERARCHICAL_ROLES,
        hierarchical_evolution=True,
        durable_dynamic_memory=True,
        memory_switches=MemorySwitches.enabled(),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._plan: Any = None
        self._committed_strategy: Mapping[str, Any] | None = None

    @property
    def persistent_plan(self) -> Any:
        return self._plan

    @property
    def committed_strategy(self) -> Mapping[str, Any] | None:
        return self._committed_strategy

    def propose_initial(self, context: AdapterContext) -> PolicyProposal:
        self._begin_run()
        self._plan = None
        self._committed_strategy = None
        orchestration = self._orchestration_payload(context)
        plan_response = self._invoke(
            MethodRole.TASK_ORCHESTRATOR,
            {
                **orchestration,
                "previous_plan": None,
                "planning_mode": "initial",
            },
        )
        result = orchestration_result_from_response(plan_response)
        self._plan = result.updated_plan
        self._committed_strategy = result.committed_strategy
        return self._generate(context, reason="initial")

    def propose_next(
        self, context: AdapterContext, feedback: CandidateFeedback
    ) -> PolicyProposal:
        orchestration = self._orchestration_payload(context)
        evidence = feedback_payload(feedback)
        if feedback.accepted:
            response = self._invoke(
                MethodRole.TASK_ORCHESTRATOR,
                {
                    **orchestration,
                    "previous_plan": self._plan,
                    "planning_mode": "advance_accepted_plan",
                    "feedback": evidence,
                },
            )
            result = orchestration_result_from_response(response)
            self._plan = result.updated_plan
            self._committed_strategy = result.committed_strategy
        else:
            manager = self._invoke(
                MethodRole.EVOLUTION_MANAGER,
                {
                    **orchestration,
                    "active_plan": self._plan,
                    "feedback": evidence,
                    "decision": "select_mutation_level",
                },
            )
            mutation = self._invoke(
                MethodRole.MUTATION_OPERATOR,
                {
                    **orchestration,
                    "active_plan": self._plan,
                    "manager_decision": manager.content,
                    "feedback": evidence,
                },
            )
            result = orchestration_result_from_response(mutation)
            self._plan = result.updated_plan
            self._committed_strategy = result.committed_strategy
        return self._generate(context, reason=feedback.kind.value)

    def _generate(
        self,
        context: AdapterContext,
        *,
        reason: str,
    ) -> PolicyProposal:
        if self._committed_strategy is None:
            raise RuntimeError("no coordination strategy was committed before coding")
        policy_payload = self._policy_payload(context, self._committed_strategy)
        response = self._invoke(
            MethodRole.POLICY_GENERATOR,
            {
                **policy_payload,
                "active_plan": self._plan,
                "generation_reason": reason,
            },
        )
        return proposal_from_response(response)

    def on_verified_rollout(
        self, rollout: RolloutRecord, context: AdapterContext
    ) -> Mapping[str, Any]:
        return self._run_memory_update(rollout, context)


class NoDynamicExperienceAdapter(FullCoEvolveAdapter):
    """Full hierarchy with every dynamic-experience call and gate disabled."""

    descriptor = MethodDescriptor(
        variant=MethodVariant.NO_DYNAMIC_EXPERIENCE,
        allowed_roles=_NO_DYNAMIC_HIERARCHICAL_ROLES,
        hierarchical_evolution=True,
        durable_dynamic_memory=False,
        memory_switches=MemorySwitches.disabled(),
    )

    def on_verified_rollout(
        self, rollout: RolloutRecord, context: AdapterContext
    ) -> Mapping[str, Any]:
        """Keep safety/audit state while making the memory ablation literal.

        The common engine still records the raw rollout, verifier result,
        accepted checkpoint, rollback state, and latest physical failure.  No
        Memory Writer model call is made and no durable experience is queried
        or promoted.
        """

        del rollout, context
        return {
            "memory_update_invoked": False,
            "strategy_promoted": False,
            "action_exemplar_promoted": False,
        }
