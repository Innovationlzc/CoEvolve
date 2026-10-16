"""Deterministic physical runtime and proposal builders for smoke tests."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Iterable, Mapping

from .contracts import (
    CheckpointRef,
    Observation,
    PhysicalDispatchResult,
    PolicyProposal,
    RobotIntent,
    RollbackResult,
    RuntimeCallbacks,
    TeamIntent,
    ValidatedPolicy,
    VerifierOutcome,
)
from .policy import IntentCompiler


def make_team_proposal(
    label: str,
    *,
    primitive: str = "cooperative_move",
    objective: str | None = None,
    mode: str = "synchronized",
    arguments: Mapping[str, Any] | None = None,
    lifecycle_phase: str = "carry",
    metadata: Mapping[str, Any] | None = None,
) -> PolicyProposal:
    """Build compiler-owned source for one deterministic paired transaction."""

    selected_objective = objective if objective is not None else label
    selected_arguments = dict(arguments or {"label": label})
    intent = TeamIntent(
        commands=(
            RobotIntent(
                "front", primitive, selected_objective, mode, selected_arguments
            ),
            RobotIntent(
                "back", primitive, selected_objective, mode, selected_arguments
            ),
        ),
        atomic=True,
        lifecycle_phase=lifecycle_phase,
        metadata=dict(metadata or {}),
    )
    return PolicyProposal(
        source=IntentCompiler().compile(intent),
        intent=intent,
        rationale=f"scripted proposal {label}",
    )


class ScriptedRuntime:
    """Small in-memory implementation of every RuntimeCallbacks port."""

    def __init__(
        self,
        outcomes: Iterable[VerifierOutcome],
        *,
        initial_observation: Observation | None = None,
        dispatch_results: Iterable[PhysicalDispatchResult] | None = None,
        rollback_succeeds: bool = True,
    ) -> None:
        self._outcomes = deque(outcomes)
        self._dispatch_results = (
            None if dispatch_results is None else deque(dispatch_results)
        )
        self._current = initial_observation or Observation(0, {"phase": "initial"})
        self._pending: Observation | None = None
        self._checkpoints: dict[str, Observation] = {}
        self._rollback_succeeds = rollback_succeeds
        self.dispatched: list[ValidatedPolicy] = []
        self.rollback_calls: list[str] = []
        self.saved_checkpoints: list[str] = []

    def observe(self) -> Observation:
        return deepcopy(self._pending or self._current)

    def dispatch(self, policy: ValidatedPolicy) -> PhysicalDispatchResult:
        self.dispatched.append(policy)
        if self._dispatch_results is not None:
            if not self._dispatch_results:
                raise AssertionError("scripted dispatch results exhausted")
            scripted = self._dispatch_results.popleft()
            if scripted.observation is not None:
                self._pending = deepcopy(scripted.observation)
            elif scripted.physically_dispatched:
                self._pending = self._next_observation(policy)
            return PhysicalDispatchResult(
                scripted.physically_dispatched,
                scripted.physical_transaction_completed,
                deepcopy(self._pending) if self._pending is not None else None,
                dict(scripted.evidence),
                scripted.error,
            )
        self._pending = self._next_observation(policy)
        return PhysicalDispatchResult(True, True, deepcopy(self._pending))

    def verify(
        self,
        observation_before: Observation,
        policy: ValidatedPolicy,
        dispatch: PhysicalDispatchResult,
    ) -> VerifierOutcome:
        if not self._outcomes:
            raise AssertionError("scripted verifier outcomes exhausted")
        outcome = self._outcomes.popleft()
        if outcome.accepted and self._pending is not None:
            self._current = self._pending
            self._pending = None
        return outcome

    def save_checkpoint(self, observation: Observation) -> CheckpointRef:
        checkpoint_id = f"checkpoint_{len(self.saved_checkpoints)}"
        self.saved_checkpoints.append(checkpoint_id)
        self._current = deepcopy(observation)
        self._pending = None
        self._checkpoints[checkpoint_id] = deepcopy(observation)
        return CheckpointRef(checkpoint_id)

    def rollback(
        self, checkpoint: CheckpointRef, expected: Observation
    ) -> RollbackResult:
        self.rollback_calls.append(checkpoint.checkpoint_id)
        if not self._rollback_succeeds:
            return RollbackResult(False, False, report="scripted rollback failure")
        restored = deepcopy(self._checkpoints[checkpoint.checkpoint_id])
        self._current = restored
        self._pending = None
        return RollbackResult(True, True, deepcopy(restored), "scripted rollback")

    def callbacks(self) -> RuntimeCallbacks:
        return RuntimeCallbacks(
            observe=self.observe,
            dispatch=self.dispatch,
            verify=self.verify,
            save_checkpoint=self.save_checkpoint,
            rollback=self.rollback,
        )

    @property
    def remaining_outcomes(self) -> int:
        return len(self._outcomes)

    def _next_observation(self, policy: ValidatedPolicy) -> Observation:
        return Observation(
            sequence=self._current.sequence + 1,
            state={
                "phase": policy.proposal.intent.lifecycle_phase,
                "dispatch_index": len(self.dispatched),
                "ast_fingerprint": policy.ast_fingerprint,
            },
        )

