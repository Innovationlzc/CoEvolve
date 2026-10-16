"""Flat-repair decision-structure contract harness (not a formal runner)."""

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
    flat_strategy_commit_from_response,
    orchestration_result_from_response,
)
from .parsing import proposal_from_response


class FlatReflectiveRepairAdapter(BaseMethodAdapter):
    """No Manager/Mutation roles and no task-level replanning after startup."""

    descriptor = MethodDescriptor(
        variant=MethodVariant.FLAT_REFLECTIVE_REPAIR,
        allowed_roles=frozenset(
            {
                MethodRole.FLAT_TASK_PLANNER,
                MethodRole.FLAT_STRATEGY_SELECTOR,
                MethodRole.FLAT_POLICY_GENERATOR,
                MethodRole.REFLECTIVE_REWRITE,
                MethodRole.MEMORY_UPDATE,
            }
        ),
        hierarchical_evolution=False,
        durable_dynamic_memory=True,
        memory_switches=MemorySwitches.enabled(),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fixed_plan: Any = None
        self._committed_strategy: Mapping[str, Any] | None = None
        self._last_policy: PolicyProposal | None = None

    @property
    def persistent_plan(self) -> Any:
        return self._fixed_plan

    @property
    def committed_strategy(self) -> Mapping[str, Any] | None:
        return self._committed_strategy

    def _select_strategy(
        self,
        context: AdapterContext,
        *,
        strategy_records: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        response = self._invoke(
            MethodRole.FLAT_STRATEGY_SELECTOR,
            {
                **self._common_payload(context),
                "fixed_task_plan": self._fixed_plan,
                "strategy_records": strategy_records,
                "selection_scope": "current_transaction_only",
                "task_plan_mutation_allowed": False,
            },
        )
        strategy = flat_strategy_commit_from_response(response)
        self._committed_strategy = strategy
        return strategy

    def propose_initial(self, context: AdapterContext) -> PolicyProposal:
        self._begin_run()
        self._fixed_plan = None
        self._committed_strategy = None
        self._last_policy = None
        # Retrieve the balanced strategy set exactly once for this physical
        # transaction.  The initial planner and current selector see the same
        # immutable snapshot; the Coder never receives strategy records.
        strategy_memory = self._strategy_memory_context(context)
        plan_response = self._invoke(
            MethodRole.FLAT_TASK_PLANNER,
            {
                **self._common_payload(context),
                **strategy_memory,
                "planning_mode": "initial_and_fixed",
            },
        )
        planned = orchestration_result_from_response(plan_response)
        self._fixed_plan = planned.updated_plan
        committed_strategy = self._select_strategy(
            context,
            strategy_records=list(strategy_memory["strategy_records"]),
        )
        policy_payload = self._policy_payload(context, committed_strategy)
        response = self._invoke(
            MethodRole.FLAT_POLICY_GENERATOR,
            {**policy_payload, "fixed_task_plan": self._fixed_plan},
        )
        self._last_policy = proposal_from_response(response)
        return self._last_policy

    def propose_next(
        self, context: AdapterContext, feedback: CandidateFeedback
    ) -> PolicyProposal:
        if self._fixed_plan is None:
            raise RuntimeError("Flat adapter requires propose_initial first")
        # A fresh, non-hierarchical commit is required on every transaction.
        # Strategy memory conditions only this selector.  Exact executable
        # exemplars are looked up after the commit and are the only memory
        # records exposed to the policy-generating role.
        strategy_memory = self._strategy_memory_context(context)
        committed_strategy = self._select_strategy(
            context,
            strategy_records=list(strategy_memory["strategy_records"]),
        )
        policy_payload = self._policy_payload(context, committed_strategy)
        if feedback.accepted:
            response = self._invoke(
                MethodRole.FLAT_POLICY_GENERATOR,
                {
                    **policy_payload,
                    "fixed_task_plan": self._fixed_plan,
                    "advance_within_fixed_plan": True,
                },
            )
        else:
            response = self._invoke(
                MethodRole.REFLECTIVE_REWRITE,
                {
                    **policy_payload,
                    "fixed_task_plan": self._fixed_plan,
                    "previous_executable": self._last_policy,
                    "verifier_feedback": feedback_payload(feedback),
                    "rewrite_scope": "executable_only",
                },
            )
        self._last_policy = proposal_from_response(response)
        return self._last_policy

    def on_verified_rollout(
        self, rollout: RolloutRecord, context: AdapterContext
    ) -> Mapping[str, Any]:
        return self._run_memory_update(rollout, context)
