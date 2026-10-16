"""Clean-room COHERENT PEFA decision contract with two executor roles."""

from __future__ import annotations

from collections import deque
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
    BaseMethodAdapter,
    MemorySwitches,
    MethodDescriptor,
    MethodRole,
    feedback_payload,
)
from .parsing import proposal_from_response


class CoherentAdaptedAdapter(BaseMethodAdapter):
    """Central proposal, paired feasibility, feedback adjustment; no hierarchy."""

    descriptor = MethodDescriptor(
        variant=MethodVariant.COHERENT_ADAPTED,
        allowed_roles=frozenset(
            {
                MethodRole.COHERENT_CENTRAL_PROPOSER,
                MethodRole.COHERENT_FRONT_FEASIBILITY,
                MethodRole.COHERENT_BACK_FEASIBILITY,
                MethodRole.COHERENT_FEEDBACK_ADJUSTER,
            }
        ),
        hierarchical_evolution=False,
        durable_dynamic_memory=False,
        memory_switches=MemorySwitches.disabled(),
    )

    def __init__(
        self,
        *args: Any,
        max_feasibility_adjustments: int = 2,
        **kwargs: Any,
    ) -> None:
        if max_feasibility_adjustments < 0:
            raise ValueError("max_feasibility_adjustments must be non-negative")
        super().__init__(*args, **kwargs)
        self._max_adjustments = max_feasibility_adjustments
        self._history: deque[Mapping[str, Any]] = deque(maxlen=5)
        self._last_proposal: PolicyProposal | None = None

    @property
    def history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._history)

    def propose_initial(self, context: AdapterContext) -> PolicyProposal | None:
        self._begin_run()
        self._history.clear()
        self._last_proposal = None
        common = self._common_payload(context)
        central = self._invoke(
            MethodRole.COHERENT_CENTRAL_PROPOSER,
            {**common, "history": [], "proposal_mode": "initial"},
        )
        candidate = proposal_from_response(central)
        self._last_proposal = self._feasibility_loop(common, candidate, None)
        return self._last_proposal

    def propose_next(
        self, context: AdapterContext, feedback: CandidateFeedback
    ) -> PolicyProposal | None:
        common = self._common_payload(context)
        if feedback.kind is FeedbackKind.GENERATION_RECOVERY:
            # Exhausting the bounded executor-feasibility dialogue does not
            # prove task failure.  Ask the central proposer for a genuinely
            # fresh child inside the common Host recovery budget.
            central = self._invoke(
                MethodRole.COHERENT_CENTRAL_PROPOSER,
                {
                    **common,
                    "history": list(self._history),
                    "proposal_mode": "generation_recovery",
                    "generation_recovery": feedback_payload(feedback),
                },
            )
            candidate = proposal_from_response(central)
            adjustment_feedback = None
        elif feedback.accepted:
            central = self._invoke(
                MethodRole.COHERENT_CENTRAL_PROPOSER,
                {
                    **common,
                    "history": list(self._history),
                    "proposal_mode": "next_subtask",
                },
            )
            candidate = proposal_from_response(central)
            adjustment_feedback: CandidateFeedback | None = None
        else:
            adjusted = self._invoke(
                MethodRole.COHERENT_FEEDBACK_ADJUSTER,
                {
                    **common,
                    "history": list(self._history),
                    "previous_proposal": self._last_proposal,
                    "feedback": feedback_payload(feedback),
                    "adjustment_source": "physical_verifier",
                },
            )
            candidate = proposal_from_response(adjusted, fallback=self._last_proposal)
            adjustment_feedback = feedback
        self._last_proposal = self._feasibility_loop(
            common, candidate, adjustment_feedback
        )
        return self._last_proposal

    def _feasibility_loop(
        self,
        common: Mapping[str, Any],
        candidate: PolicyProposal,
        feedback: CandidateFeedback | None,
    ) -> PolicyProposal | None:
        current = candidate
        for adjustment_index in range(self._max_adjustments + 1):
            front = self._invoke(
                MethodRole.COHERENT_FRONT_FEASIBILITY,
                {
                    **common,
                    "candidate": current,
                    "actor": "front",
                    "history": list(self._history),
                },
            )
            back = self._invoke(
                MethodRole.COHERENT_BACK_FEASIBILITY,
                {
                    **common,
                    "candidate": current,
                    "actor": "back",
                    "history": list(self._history),
                },
            )
            front_ok = _is_feasible(front.content)
            back_ok = _is_feasible(back.content)
            if front_ok and back_ok:
                return current
            if adjustment_index >= self._max_adjustments:
                # The common Host recovery loop decides whether another fresh
                # generation cycle exists and, only after all cycles fail,
                # whether a SYSTEM_STALL ledger can be emitted.
                return None
            adjusted = self._invoke(
                MethodRole.COHERENT_FEEDBACK_ADJUSTER,
                {
                    **common,
                    "candidate": current,
                    "front_feedback": front.content,
                    "back_feedback": back.content,
                    "physical_feedback": None
                    if feedback is None
                    else feedback_payload(feedback),
                    "adjustment_source": "executor_feasibility",
                },
            )
            current = proposal_from_response(adjusted, fallback=current)
        return None

    def on_verified_rollout(
        self, rollout: RolloutRecord, context: AdapterContext
    ) -> Mapping[str, Any]:
        item = {
            "transaction_index": rollout.transaction_index,
            "checkpoint_id": rollout.checkpoint.checkpoint_id,
            "accepted": rollout.verifier.accepted,
            "failure_code": rollout.verifier.failure_code,
            "report": rollout.verifier.report,
        }
        self._history.append(item)
        return {
            "pefa_history_entry": item,
            "history_size": len(self._history),
            "durable_promotion": False,
        }


def _is_feasible(content: Any) -> bool:
    return isinstance(content, Mapping) and content.get("feasible") is True
