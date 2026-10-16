"""Clean-room REMAC decision-structure contract harness (smoke only)."""

from __future__ import annotations

from typing import Any, Mapping

from coevolve_experiments.contracts import MethodVariant
from coevolve_experiments.runtime.contracts import (
    CandidateFeedback,
    FeedbackKind,
    PolicyProposal,
    RolloutRecord,
)

from .base import (
    AdapterContext,
    AdapterContractError,
    BaseMethodAdapter,
    MemorySwitches,
    MethodDescriptor,
    MethodRole,
    feedback_payload,
)
from .parsing import proposal_from_response


class RemacAdaptedAdapter(BaseMethodAdapter):
    """Pre/post-condition reflection with bounded task-level replanning."""

    descriptor = MethodDescriptor(
        variant=MethodVariant.REMAC_ADAPTED,
        allowed_roles=frozenset(
            {
                MethodRole.REMAC_PLANNER,
                MethodRole.REMAC_PRECONDITION_REFLECTION,
                MethodRole.REMAC_POSTCONDITION_REFLECTION,
                MethodRole.REMAC_REPLANNER,
            }
        ),
        hierarchical_evolution=False,
        durable_dynamic_memory=False,
        memory_switches=MemorySwitches.disabled(),
    )

    def __init__(
        self,
        *args: Any,
        max_postcondition_retries: int = 2,
        **kwargs: Any,
    ) -> None:
        if max_postcondition_retries < 0:
            raise ValueError("max_postcondition_retries must be non-negative")
        super().__init__(*args, **kwargs)
        self._max_retries = max_postcondition_retries
        self._consecutive_retries = 0
        self._task_plan: Any = None
        self._run_reflections: list[Mapping[str, Any]] = []

    @property
    def run_local_reflections(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._run_reflections)

    def propose_initial(self, context: AdapterContext) -> PolicyProposal:
        self._begin_run()
        self._consecutive_retries = 0
        self._task_plan = None
        self._run_reflections.clear()
        common = self._common_payload(context)
        plan_response = self._invoke(
            MethodRole.REMAC_PLANNER,
            {
                **common,
                "replan_scope": "full_task",
                "run_local_reflections": [],
            },
        )
        self._task_plan = plan_response.content
        return self._precondition_check(common, plan_response)

    def propose_next(
        self, context: AdapterContext, feedback: CandidateFeedback
    ) -> PolicyProposal | None:
        common = self._common_payload(context)
        if feedback.kind is FeedbackKind.GENERATION_RECOVERY:
            # A method-local reflection cap is not a physical or task failure.
            # Hand control back to REMAC's planner under the Host's common
            # generation-recovery cycle instead of terminating the run early.
            plan_response = self._invoke(
                MethodRole.REMAC_PLANNER,
                {
                    **common,
                    "previous_task_plan": self._task_plan,
                    "replan_scope": "full_task_generation_recovery",
                    "run_local_reflections": list(self._run_reflections),
                    "generation_recovery": feedback_payload(feedback),
                },
            )
            self._consecutive_retries = 0
        elif feedback.accepted:
            self._consecutive_retries = 0
            plan_response = self._invoke(
                MethodRole.REMAC_PLANNER,
                {
                    **common,
                    "previous_task_plan": self._task_plan,
                    "replan_scope": "remaining_task",
                    "run_local_reflections": list(self._run_reflections),
                },
            )
        else:
            consumes_physical_retry = feedback.kind is FeedbackKind.VERIFIED_REJECTED
            if (
                consumes_physical_retry
                and self._consecutive_retries >= self._max_retries
            ):
                # ``None`` is a typed request to the common Host recovery
                # loop.  It is not itself a SYSTEM_STALL or task failure.
                return None
            post = self._invoke(
                MethodRole.REMAC_POSTCONDITION_REFLECTION,
                {
                    **common,
                    "task_plan": self._task_plan,
                    "feedback": feedback_payload(feedback),
                    "retry_index": self._consecutive_retries,
                    "consumes_physical_retry": consumes_physical_retry,
                },
            )
            post_record = {
                "kind": "postcondition",
                "content": post.content,
                "feedback": feedback_payload(feedback),
            }
            self._run_reflections.append(post_record)
            plan_response = self._invoke(
                MethodRole.REMAC_REPLANNER,
                {
                    **common,
                    "previous_task_plan": self._task_plan,
                    "postcondition_reflection": post.content,
                    "replan_scope": "remaining_or_full_task",
                    "run_local_reflections": list(self._run_reflections),
                },
            )
            if consumes_physical_retry:
                self._consecutive_retries += 1
        self._task_plan = plan_response.content
        return self._precondition_check(common, plan_response)

    def _precondition_check(
        self, common: Mapping[str, Any], plan_response: Any
    ) -> PolicyProposal:
        plan_candidate: PolicyProposal | None = None
        try:
            plan_candidate = proposal_from_response(plan_response)
        except AdapterContractError:
            pass
        reflection = self._invoke(
            MethodRole.REMAC_PRECONDITION_REFLECTION,
            {
                **common,
                "task_plan": self._task_plan,
                "check": "preconditions_and_parallel_dependencies",
                "run_local_reflections": list(self._run_reflections),
            },
        )
        proposal = proposal_from_response(reflection, fallback=plan_candidate)
        self._run_reflections.append(
            {"kind": "precondition", "content": reflection.content}
        )
        return proposal

    def on_verified_rollout(
        self, rollout: RolloutRecord, context: AdapterContext
    ) -> Mapping[str, Any]:
        record = {
            "kind": "verified_rollout",
            "accepted": rollout.verifier.accepted,
            "failure_code": rollout.verifier.failure_code,
            "report": rollout.verifier.report,
            "checkpoint_id": rollout.checkpoint.checkpoint_id,
        }
        self._run_reflections.append(record)
        return {"run_local_reflection": record, "durable_promotion": False}
