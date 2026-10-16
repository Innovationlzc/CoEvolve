"""Deterministic host-side control for policy evolution."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Literal

from .contracts import (
    EvaluationReport,
    ExecutionTrace,
    PolicyScript,
    RobotPolicyScript,
    write_json,
)
from .failure_taxonomy import classify_report
from .coordination_specs import (
    COORDINATED_LINEAR_DISTANCE_CHOICES_M,
    COORDINATION_PRIMITIVE_METHODS,
    LOADED_TRANSLATION_DIRECTIONS,
    coordination_relationship_can_change,
    coordination_role_variant_limit,
)
from .transition_verifier import (
    TransitionOutcome,
    transition_outcome_from_report,
)


ControllerAction = Literal[
    "task_complete",
    "switch_strategy",
    "change_coordination",
    "rewrite_script",
    "tune_parameter",
    "continue",
    "repair_system",
    "stop_budget_exhausted",
]


@dataclass(frozen=True)
class EvolutionAttempt:
    """One verifier-evaluated script and its host-controller outcome."""

    record_index: int
    policy_attempt: int
    script_id: str
    checkpoint_id: str
    strategy_signature: str
    motion_objective_signature: str
    role_signature: str
    parameter_signature: str
    step_accepted: bool
    task_complete: bool
    failure_type: str
    failure_level: str
    transition_outcome: str
    physically_dispatched: bool
    counts_toward_budget: bool
    physical_iteration: int
    strategy_failure_count: int = 0
    semantic_failure_count: int = 0
    route_stall_iterations: int = 0
    strategy_failure_budget: int = 0
    strategy_evidence_active: bool = True
    controller_action: ControllerAction = "continue"
    should_evolve: bool = True
    budget_remaining: int = 0
    hard_iteration_limit_reached: bool = False

    @property
    def evidence_key(self) -> str:
        return f"{self.checkpoint_id}:{self.policy_attempt}:{self.script_id}"


@dataclass(frozen=True)
class EvolutionHistoryView:
    """Read-only evidence supplied to the deterministic controller."""

    attempts: tuple[EvolutionAttempt, ...]
    max_physical_iterations: int
    strategy_failure_budget: int
    hard_iteration_limit_reached: bool = False

    @property
    def current(self) -> EvolutionAttempt:
        if not self.attempts:
            raise ValueError("evolution history must contain the current attempt")
        return self.attempts[-1]

    @property
    def physical_iterations(self) -> int:
        return sum(1 for item in self.attempts if item.counts_toward_budget)


@dataclass(frozen=True)
class EvolutionDecision:
    """Host lifecycle decision plus the effective model-selected repair scope."""

    action: ControllerAction
    should_evolve: bool
    repair_level: str
    reason: str
    required_change: str
    failure_type: str
    failure_level: str
    strategy_signature: str
    strategy_failure_count: int
    strategy_failure_budget: int
    physical_iterations: int
    max_physical_iterations: int
    budget_remaining: int
    budget_exhausted: bool
    semantic_failure_count: int = 0
    route_stall_iterations: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class EvolutionController:
    """Own task completion, iteration budgets, and engineering repair defaults."""

    max_physical_iterations: int = 20
    strategy_failure_budget: int = 0
    output_dir: Path | None = None
    resume_history: bool = True
    history_dirs: tuple[Path, ...] = ()
    _attempts: list[EvolutionAttempt] = field(default_factory=list, init=False, repr=False)
    _active_checkpoint_id: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.max_physical_iterations = int(self.max_physical_iterations)
        self.strategy_failure_budget = int(self.strategy_failure_budget)
        if self.max_physical_iterations < 0:
            raise ValueError("max_physical_iterations must be non-negative")
        if self.strategy_failure_budget < 0:
            raise ValueError("strategy_failure_budget must be non-negative")
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        self.history_dirs = tuple(Path(path) for path in self.history_dirs)
        if self.resume_history:
            self._load_history()

    @property
    def attempts(self) -> tuple[EvolutionAttempt, ...]:
        return tuple(self._attempts)

    @property
    def physical_iterations(self) -> int:
        return sum(1 for item in self._attempts if item.counts_toward_budget)

    def rewind_strategy_evidence_to_checkpoint(
        self,
        checkpoint_id: str,
        *,
        release_checkpoint_failures: bool = False,
    ) -> int:
        """Retire failed branch evidence after an explicit checkpoint rewind.

        An operator-supplied branch lesson opens a fresh strategy search at the
        restored ancestor. In that case, old failures at the ancestor may come
        from the retired controller/primitive revision and must remain in the
        physical-iteration accounting without continuing to block candidates.
        """

        checkpoint_id = str(checkpoint_id)
        self._active_checkpoint_id = checkpoint_id
        changed = 0
        updated: list[EvolutionAttempt] = []
        for item in self._attempts:
            active = bool(
                item.step_accepted
                or item.failure_level == "system"
                or (
                    item.checkpoint_id == checkpoint_id
                    and not release_checkpoint_failures
                )
            )
            if active != item.strategy_evidence_active:
                changed += 1
            updated.append(replace(item, strategy_evidence_active=active))
        self._attempts = updated
        return changed

    def set_active_checkpoint(self, checkpoint_id: str) -> None:
        """Select the trusted state whose physical failures constrain the next action."""

        checkpoint_id = str(checkpoint_id).strip()
        if checkpoint_id:
            self._active_checkpoint_id = checkpoint_id

    @property
    def active_checkpoint_id(self) -> str:
        """Return the checkpoint-local physical-evidence scope."""

        if self._active_checkpoint_id:
            return self._active_checkpoint_id
        return self._attempts[-1].checkpoint_id if self._attempts else ""

    @property
    def strategy_failure_counts(self) -> dict[str, int]:
        """Return failures for each primitive and semantic-objective budget."""

        checkpoint_id = self.active_checkpoint_id
        attempts = tuple(
            item
            for item in self._attempts
            if not checkpoint_id or item.checkpoint_id == checkpoint_id
        )
        identities = sorted(
            {
                (item.strategy_signature, item.motion_objective_signature)
                for item in attempts
                if item.strategy_signature and item.strategy_evidence_active
            }
        )
        return {
            _strategy_budget_key(strategy, objective): count
            for strategy, objective in identities
            if (
                count := self._strategy_failure_count(
                    attempts,
                    strategy_signature=strategy,
                    motion_objective_signature=objective,
                    checkpoint_id=checkpoint_id,
                )
            )
            > 0
        }

    @property
    def exhausted_strategy_signatures(self) -> tuple[str, ...]:
        """Return no count-blocked strategies.

        Rejected executable ASTs remain protected separately. Repeating a semantic
        strategy is never forbidden merely because it failed a fixed number of times.
        """

        return ()

    @property
    def exhausted_motion_objectives(self) -> tuple[str, ...]:
        """Return no count-blocked semantic objectives."""

        return ()

    def decide_next_step(self, history: EvolutionHistoryView) -> EvolutionDecision:
        """Stop or continue safely and provide the engineering repair default.

        In formal model-role runs, the host-only fields are withheld and the
        Evolution Manager replaces this default with its own mutation level.
        """

        current = history.current
        route_progress_outcomes = {
            TransitionOutcome.ACCEPT_PROGRESS.value,
            TransitionOutcome.TASK_COMPLETE.value,
        }
        current_is_route_progress = current.transition_outcome in route_progress_outcomes
        strategy_previously_accepted = any(
            item is not current
            and item.counts_toward_budget
            and item.transition_outcome in route_progress_outcomes
            and item.strategy_signature == current.strategy_signature
            and item.motion_objective_signature == current.motion_objective_signature
            for item in history.attempts
        )
        coordination_roles_exhausted = _coordination_roles_exhausted(
            history.attempts,
            current,
        )

        if current.task_complete:
            return self._make_decision(
                history,
                action="task_complete",
                should_evolve=False,
                repair_level="none",
                reason="The physics verifier confirmed the full task completion contract.",
                required_change="none",
            )

        if history.hard_iteration_limit_reached or (
            history.max_physical_iterations > 0
            and history.physical_iterations >= history.max_physical_iterations
        ):
            return self._make_decision(
                history,
                action="stop_budget_exhausted",
                should_evolve=False,
                repair_level="none",
                reason=(
                    "The host evolution budget is exhausted; generating another script would "
                    "create an unexecutable extra attempt."
                ),
                required_change="stop and report the best verified state",
                budget_exhausted=True,
            )

        if current.failure_level == "system":
            externally_blocked = current.failure_type in {
                "runtime_state_missing",
                "manual_review_required",
            }
            return self._make_decision(
                history,
                action="repair_system",
                should_evolve=not externally_blocked,
                repair_level="system",
                reason=(
                    "The failure is infrastructure/evidence related and cannot evaluate a "
                    "physical cooperation strategy."
                ),
                required_change=(
                    "restore required runtime evidence before another rollout"
                    if externally_blocked
                    else "repair generated code or the execution preflight without changing strategy"
                ),
            )

        if current.step_accepted:
            return self._make_decision(
                history,
                action="continue",
                should_evolve=True,
                repair_level="none",
                reason=(
                    "The transition made measured connected-route progress."
                    if current_is_route_progress
                    else (
                        "The endpoint is a restorable setup checkpoint, but it did not "
                        "complete the active route subtask."
                    )
                ),
                required_change=(
                    "select the next bounded subtask from current verified state"
                    if current_is_route_progress
                    else (
                        "continue the active subtask from the new setup checkpoint; do "
                        "not advance the route phase or treat setup as strategy success"
                    )
                ),
            )

        if current.semantic_failure_count >= 2:
            current_method = current.strategy_signature.rsplit(":", 1)[-1]
            coordination_change_available = coordination_relationship_can_change(
                current_method,
                current.strategy_signature.split(":", 1)[0]
                if ":" in current.strategy_signature
                else "sync",
            )
            return self._make_decision(
                history,
                action=(
                    "change_coordination"
                    if coordination_change_available
                    else "switch_strategy"
                ),
                should_evolve=True,
                repair_level=(
                    "coordination" if coordination_change_available else "strategy"
                ),
                reason=(
                    "The same checkpoint-local semantic strategy, role pattern, and "
                    "canonical failure class have now been physically rejected twice."
                ),
                required_change=(
                    "parameter_tuning is forbidden for the next candidate; change the "
                    "coordination relationship or the strategy structure without the host "
                    "prescribing a replacement direction"
                ),
            )

        if current.failure_level == "script":
            structural_physical_failure = current.failure_type in {
                "formation_desync",
                "payload_drop",
                "payload_not_supported",
                "payload_slip",
                "payload_wall_contact",
                "robot_fall",
            }
            return self._make_decision(
                history,
                action="rewrite_script",
                should_evolve=True,
                repair_level="script",
                reason=(
                    "The physics verifier traced the failure to the generated executable "
                    "transaction, not to a mandatory role-switch policy."
                ),
                required_change=(
                    (
                        "identify the bad API call, parameter, or front/back command relation "
                        "and treat the failed support/coordination transaction as a code defect; "
                        "emit a materially different executable AST by changing its body-frame "
                        "objective, coordination primitive/mode, or front/back mover-role "
                        "structure. A same-objective numerical rewrite (especially a larger "
                        "distance, speed, duration, or angle) is forbidden"
                    )
                    if structural_physical_failure
                    else (
                        "identify the bad API call, parameter, or front/back command relation "
                        "and its measured physical consequence; then emit a materially "
                        "different executable AST. Do not repair by changing only identifiers, "
                        "phase labels, roles, rationale, or other metadata"
                    )
                ),
            )

        if _linear_parameter_envelope_exhausted(current):
            return self._make_decision(
                history,
                action="switch_strategy",
                should_evolve=True,
                repair_level="strategy",
                reason=(
                    "The rejected loaded linear transaction already used the minimum "
                    "supported distance tier, so another parameter-only repair is not "
                    "executable."
                ),
                required_change=(
                    "preserve the trusted checkpoint and payload-safety contract, but "
                    "change the semantic motion objective or maneuver primitive before "
                    "another physical dispatch"
                ),
            )

        if current.failure_type == "straightness_regression":
            return self._make_decision(
                history,
                action="switch_strategy",
                should_evolve=True,
                repair_level="strategy",
                reason=(
                    "A non-turn transport left its persistent straight/lateral axis."
                ),
                required_change=(
                    "keep the restored checkpoint stationary and execute exactly the "
                    "verifier-reported heading repair before another translation: a "
                    "synchronized in-place turn for shared bias or relative yaw alignment "
                    "for a front/back heading split"
                ),
            )

        if current.failure_level == "strategy":
            return self._make_decision(
                history,
                action="switch_strategy",
                should_evolve=True,
                repair_level="strategy",
                reason="The verifier classified the failure at the strategy layer.",
                required_change=(
                    "change semantic motion objective and/or coordination mode/mover method; "
                    "parameter-only retries are forbidden"
                ),
            )

        current_method = current.strategy_signature.rsplit(":", 1)[-1]
        if (
            current.failure_level == "coordination"
            and current_method
            in {
                "coordinated_arc_transport",
                "coordinated_pivot",
            }
            and current.failure_type
            in {
                "payload_drop",
                "payload_not_supported",
                "payload_slip",
                "payload_unstable",
                "robot_fall",
            }
        ):
            turn_parameters = _shared_turn_parameters(current)
            if (
                current_method
                in {"coordinated_arc_transport", "coordinated_pivot"}
                and turn_parameters is not None
                and turn_parameters[0] <= 5.0 + 1.0e-9
            ):
                return self._make_decision(
                    history,
                    action="change_coordination",
                    should_evolve=True,
                    repair_level="coordination",
                    reason=(
                        "The loaded arc/pivot still lost support at the minimum safe angle; "
                        "extending the SONIC motion window would increase total travel exposure."
                    ),
                    required_change=(
                        "keep the observed route objective but change the temporary inner/outer "
                        "relationship, body-frame travel direction, or coordination primitive; "
                        "do not increase duration_s"
                    ),
                )
            return self._make_decision(
                history,
                action="tune_parameter",
                should_evolve=True,
                repair_level="execution",
                reason=(
                    "A loaded continuous turn lost physical support; reduce the per-step turn "
                    "load before changing a model-selected direction or inner/outer relation."
                ),
                required_change=_execution_repair_required_change(
                    current,
                    fallback=(
                        "keep turn direction and roles fixed; halve the bounded angle while "
                        "preserving duration"
                    ),
                ),
            )

        if (
            current.failure_level == "coordination"
            and strategy_previously_accepted
            and current.failure_type
            in {
                "formation_desync",
                "payload_not_supported",
                "payload_slip",
                "payload_unstable",
            }
        ):
            return self._make_decision(
                history,
                action="tune_parameter",
                should_evolve=True,
                repair_level="execution",
                reason=(
                    "The same strategy has a prior physics-accepted motion rollout in this "
                    "stage; this failure probes its execution envelope, not a new role strategy."
                ),
                required_change=(
                    _execution_repair_required_change(
                        current,
                        fallback=(
                            "keep the verified mode, mover, and role pattern; reduce one bounded "
                            "execution magnitude before considering a coordination switch"
                        ),
                    )
                ),
            )

        if (
            current.failure_level == "coordination"
            and not _strategy_has_coordination_repair(current.strategy_signature)
        ):
            return self._make_decision(
                history,
                action="tune_parameter",
                should_evolve=True,
                repair_level="execution",
                reason=(
                    "The selected primitive exposes no alternate coordination mode or "
                    "temporary role assignment, so its remaining safe repair is the bounded "
                    "execution envelope."
                ),
                required_change=(
                    _execution_repair_required_change(
                        current,
                        fallback=(
                            "keep the primitive and motion objective; reduce exactly one bounded "
                            "execution magnitude"
                        ),
                    )
                ),
            )

        if current.failure_level == "coordination" and coordination_roles_exhausted:
            return self._make_decision(
                history,
                action="tune_parameter",
                should_evolve=True,
                repair_level="execution",
                reason=(
                    "All finite temporary role permutations for this coordination primitive "
                    "and motion objective have physical rejection evidence."
                ),
                required_change=(
                    _execution_repair_required_change(
                        current,
                        fallback=(
                            "keep the selected role assignment and motion objective; reduce exactly "
                            "one bounded execution magnitude"
                        ),
                    )
                ),
            )

        if current.failure_level == "coordination":
            return self._make_decision(
                history,
                action="change_coordination",
                should_evolve=True,
                repair_level="coordination",
                reason="The motion objective is not disproven, but the robot relationship failed.",
                required_change=(
                    "keep the mover objective and change sync/async mode or temporary support/mover roles"
                ),
            )

        return self._make_decision(
            history,
            action="tune_parameter",
            should_evolve=True,
            repair_level="execution",
            reason="The strategy has not been disproven and execution needs one bounded adjustment.",
            required_change=(
                _execution_repair_required_change(
                    current,
                    fallback=(
                        "keep strategy and role pattern; tune exactly one bounded execution parameter"
                    ),
                )
            ),
        )

    @staticmethod
    def _make_decision(
        history: EvolutionHistoryView,
        *,
        action: ControllerAction,
        should_evolve: bool,
        repair_level: str,
        reason: str,
        required_change: str,
        budget_exhausted: bool = False,
    ) -> EvolutionDecision:
        current = history.current
        physical_iterations = history.physical_iterations
        unlimited = history.max_physical_iterations == 0
        remaining = (
            -1
            if unlimited
            else max(0, history.max_physical_iterations - physical_iterations)
        )
        return EvolutionDecision(
            action=action,
            should_evolve=should_evolve,
            repair_level=repair_level,
            reason=reason,
            required_change=required_change,
            failure_type=current.failure_type,
            failure_level=current.failure_level,
            strategy_signature=current.strategy_signature,
            strategy_failure_count=current.strategy_failure_count,
            strategy_failure_budget=history.strategy_failure_budget,
            physical_iterations=physical_iterations,
            max_physical_iterations=history.max_physical_iterations,
            budget_remaining=remaining,
            budget_exhausted=bool(budget_exhausted or (not unlimited and remaining == 0)),
            semantic_failure_count=current.semantic_failure_count,
            route_stall_iterations=current.route_stall_iterations,
        )

    def record_and_decide(
        self,
        *,
        policy: PolicyScript,
        report: EvaluationReport,
        trace: ExecutionTrace | None,
        checkpoint_id: str,
        physically_dispatched: bool | None = None,
        hard_iteration_limit_reached: bool = False,
        strategy_failure_count: int | None = None,
    ) -> EvolutionDecision:
        """Record one evaluated script and persist its deterministic decision."""

        if physically_dispatched is None:
            physically_dispatched = trace_has_physical_dispatch(trace)
        classification = classify_report(report)
        transition_outcome = transition_outcome_from_report(report).value
        signature = policy_strategy_signature(policy)
        motion_objective = policy_motion_objective_signature(policy)
        counts_toward_budget = bool(physically_dispatched)
        existing = self._find_existing(policy, checkpoint_id=checkpoint_id)
        prior_attempts = [item for item in self._attempts if item is not existing]
        physical_iteration = sum(
            1 for item in prior_attempts if item.counts_toward_budget
        ) + (1 if counts_toward_budget else 0)
        draft = EvolutionAttempt(
            record_index=len(prior_attempts) + 1,
            policy_attempt=int(policy.attempt),
            script_id=policy.script_id,
            checkpoint_id=str(checkpoint_id),
            strategy_signature=signature,
            motion_objective_signature=motion_objective,
            role_signature=policy_role_signature(policy),
            parameter_signature=policy_parameter_signature(policy),
            step_accepted=bool(report.success),
            task_complete=bool(report.metrics.get("task_complete", 0.0)),
            failure_type=str(report.failure_type),
            failure_level=classification.level,
            transition_outcome=transition_outcome,
            physically_dispatched=bool(physically_dispatched),
            counts_toward_budget=counts_toward_budget,
            physical_iteration=physical_iteration,
            strategy_failure_budget=self.strategy_failure_budget,
            hard_iteration_limit_reached=bool(hard_iteration_limit_reached),
        )
        computed_failure_count = self._strategy_failure_count(
            (*prior_attempts, draft),
            strategy_signature=signature,
            motion_objective_signature=motion_objective,
            checkpoint_id=str(checkpoint_id),
        )
        draft = replace(
            draft,
            strategy_failure_count=max(
                computed_failure_count,
                int(strategy_failure_count or 0),
            ),
            semantic_failure_count=self._semantic_failure_count(
                (*prior_attempts, draft),
                current=draft,
            ),
            route_stall_iterations=self._route_stall_iterations(
                (*prior_attempts, draft),
            ),
        )
        view = EvolutionHistoryView(
            attempts=(*prior_attempts, draft),
            max_physical_iterations=self.max_physical_iterations,
            strategy_failure_budget=self.strategy_failure_budget,
            hard_iteration_limit_reached=bool(hard_iteration_limit_reached),
        )
        decision = self.decide_next_step(view)
        completed = replace(
            draft,
            controller_action=decision.action,
            should_evolve=decision.should_evolve,
            budget_remaining=decision.budget_remaining,
        )
        self._attempts = [*prior_attempts, completed]
        self._persist(decision)
        return decision

    def escalate_candidate_space_exhausted(
        self,
        decision: EvolutionDecision,
        *,
        error: str,
    ) -> EvolutionDecision:
        """Keep generation moving when no host-valid candidate can be compiled."""

        if not decision.should_evolve or decision.repair_level not in {
            "none",
            "coordination",
            "script",
            "execution",
            "strategy",
        }:
            raise ValueError(
                "candidate-space escalation requires an active continuation, script, execution, "
                "coordination, or strategy repair"
            )
        coordination_escalation = decision.repair_level == "coordination"
        escalated = replace(
            decision,
            action="switch_strategy",
            repair_level="strategy",
            reason=(
                "The structured parameter stage exhausted its bounded model retries without "
                f"producing a host-valid role relation: {error}"
            ),
            required_change=(
                "leave the coordination repair layer and select a different semantic motion "
                "objective and/or coordination primitive; do not retry the exhausted role transaction"
                if coordination_escalation
                else (
                    "select another non-exhausted semantic motion objective or coordination "
                    "primitive; do not retry the rejected generated transaction"
                )
            ),
        )
        if self._attempts:
            self._attempts[-1] = replace(
                self._attempts[-1],
                controller_action=escalated.action,
                should_evolve=escalated.should_evolve,
            )
            self._persist(escalated)
        return escalated

    def metrics(self) -> dict[str, object]:
        """Return aggregate-ready evolution efficiency and strategy evidence."""

        physical = [item for item in self._attempts if item.counts_toward_budget]
        route_progress_outcomes = {
            TransitionOutcome.ACCEPT_PROGRESS.value,
            TransitionOutcome.TASK_COMPLETE.value,
        }
        completion_index = next(
            (index for index, item in enumerate(physical, start=1) if item.task_complete),
            None,
        )
        task_success = completion_index is not None
        first_failure = next((item for item in physical if not item.step_accepted), None)
        completion_record = physical[completion_index - 1] if completion_index else None
        successful_strategies = [
            item
            for item in physical[:completion_index]
            if completion_index
            and item.transition_outcome in route_progress_outcomes
            and item.strategy_signature
        ]
        final_strategy_record = successful_strategies[-1] if successful_strategies else completion_record
        signatures = [item.strategy_signature for item in physical if item.strategy_signature]
        strategy_switch_count = sum(
            1 for previous, current in zip(signatures, signatures[1:]) if previous != current
        )
        failure_driven_switches = sum(
            1
            for previous, current in zip(physical, physical[1:])
            if (
                not previous.step_accepted
                and previous.strategy_signature
                and current.strategy_signature
                and previous.strategy_signature != current.strategy_signature
            )
        )
        parameter_only_retries = sum(
            1
            for previous, current in zip(physical, physical[1:])
            if (
                not previous.step_accepted
                and previous.strategy_signature
                and previous.strategy_signature == current.strategy_signature
                and previous.role_signature == current.role_signature
                and previous.parameter_signature != current.parameter_signature
            )
        )
        initial_strategy = first_failure.strategy_signature if first_failure else ""
        final_strategy = final_strategy_record.strategy_signature if final_strategy_record else ""
        verified_improvement = bool(
            task_success
            and initial_strategy
            and final_strategy
            and initial_strategy != final_strategy
            and failure_driven_switches > 0
        )
        iterations = len(physical)
        accepted_iterations = sum(1 for item in physical if item.step_accepted)
        setup_iterations = sum(
            1
            for item in physical
            if item.transition_outcome == TransitionOutcome.ACCEPT_SETUP.value
        )
        progress_iterations = sum(
            1 for item in physical if item.transition_outcome in route_progress_outcomes
        )
        current_route_stall_iterations = self._route_stall_iterations(tuple(physical))
        maximum_route_stall_iterations = max(
            (item.route_stall_iterations for item in physical),
            default=0,
        )
        rejected_iterations = iterations - accepted_iterations
        stopped_by_budget = any(
            item.controller_action == "stop_budget_exhausted" for item in self._attempts
        )
        unlimited = self.max_physical_iterations == 0
        return {
            "task_success": task_success,
            "task_success_rate": 1.0 if task_success else 0.0,
            "physical_iterations": iterations,
            "accepted_physical_iterations": accepted_iterations,
            "accepted_setup_iterations": setup_iterations,
            "accepted_route_progress_iterations": progress_iterations,
            "current_route_stall_iterations": current_route_stall_iterations,
            "maximum_route_stall_iterations": maximum_route_stall_iterations,
            "rejected_physical_iterations": rejected_iterations,
            "physical_accounting_complete": (
                accepted_iterations + rejected_iterations == iterations
            ),
            "iterations_to_success": completion_index,
            "evolution_efficiency": (1.0 / completion_index) if completion_index else 0.0,
            "budget": {
                "maximum_physical_iterations": (
                    None if unlimited else self.max_physical_iterations
                ),
                "unlimited": unlimited,
                "used": iterations,
                "remaining": (
                    None
                    if unlimited
                    else max(0, self.max_physical_iterations - iterations)
                ),
                "exhausted": bool(
                    stopped_by_budget
                    or (
                        not unlimited
                        and iterations >= self.max_physical_iterations
                        and not task_success
                    )
                ),
            },
            "strategy_improvement": {
                "initial_failed_strategy": initial_strategy or None,
                "initial_failure_type": first_failure.failure_type if first_failure else None,
                "initial_failure_level": first_failure.failure_level if first_failure else None,
                "final_successful_strategy": final_strategy or None,
                "distinct_strategy_count": len(set(signatures)),
                "strategy_switch_count": strategy_switch_count,
                "failure_driven_strategy_switch_count": failure_driven_switches,
                "parameter_only_retry_count": parameter_only_retries,
                "verified_strategy_improvement": verified_improvement,
            },
        }

    def _find_existing(
        self,
        policy: PolicyScript,
        *,
        checkpoint_id: str,
    ) -> EvolutionAttempt | None:
        key = f"{checkpoint_id}:{policy.attempt}:{policy.script_id}"
        return next((item for item in self._attempts if item.evidence_key == key), None)

    @staticmethod
    def _strategy_failure_count(
        attempts: tuple[EvolutionAttempt, ...],
        *,
        strategy_signature: str,
        motion_objective_signature: str = "",
        checkpoint_id: str = "",
    ) -> int:
        if not strategy_signature:
            return 0
        physical = [
            item
            for item in attempts
            if item.counts_toward_budget
            and item.strategy_evidence_active
            and (not checkpoint_id or item.checkpoint_id == checkpoint_id)
        ]
        last_success_index = max(
            (
                index
                for index, item in enumerate(physical)
                if item.transition_outcome
                in {
                    TransitionOutcome.ACCEPT_PROGRESS.value,
                    TransitionOutcome.TASK_COMPLETE.value,
                }
                and item.strategy_signature == strategy_signature
                and item.motion_objective_signature == motion_objective_signature
            ),
            default=-1,
        )
        return sum(
            1
            for item in physical[last_success_index + 1 :]
            if item.strategy_signature == strategy_signature
            and item.motion_objective_signature == motion_objective_signature
            and not item.step_accepted
            and item.failure_level != "system"
        )

    @staticmethod
    def _semantic_failure_count(
        attempts: tuple[EvolutionAttempt, ...],
        *,
        current: EvolutionAttempt,
    ) -> int:
        """Count matching semantic failures while ignoring numerical parameters."""

        if (
            current.transition_outcome
            in {
                TransitionOutcome.ACCEPT_PROGRESS.value,
                TransitionOutcome.TASK_COMPLETE.value,
            }
            or current.failure_level == "system"
            or not current.strategy_signature
        ):
            return 0
        canonical_failure = _canonical_failure_class(current.failure_type)
        return sum(
            1
            for item in attempts
            if item.counts_toward_budget
            and item.strategy_evidence_active
            and item.checkpoint_id == current.checkpoint_id
            and item.strategy_signature == current.strategy_signature
            and item.motion_objective_signature
            == current.motion_objective_signature
            and item.role_signature == current.role_signature
            and _canonical_failure_class(item.failure_type) == canonical_failure
            and not item.step_accepted
            and item.failure_level != "system"
        )

    @staticmethod
    def _route_stall_iterations(
        attempts: tuple[EvolutionAttempt, ...],
    ) -> int:
        """Count physical attempts since the latest measured route progress.

        Restorable setup checkpoints deliberately do not reset this evidence.
        Provider/runtime failures and pre-dispatch rejections neither increment nor
        reset it because they did not evaluate a task strategy in physics.
        """

        count = 0
        for item in reversed(attempts):
            if not item.counts_toward_budget:
                continue
            if item.transition_outcome in {
                TransitionOutcome.ACCEPT_PROGRESS.value,
                TransitionOutcome.TASK_COMPLETE.value,
            }:
                break
            if item.transition_outcome == TransitionOutcome.SYSTEM_INVALID.value:
                continue
            count += 1
        return count

    def _persist(self, decision: EvolutionDecision) -> None:
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        current = self._attempts[-1]
        write_json(
            self.output_dir
            / f"evolution_controller_decision_{current.policy_attempt:02d}.json",
            {
                "attempt": asdict(current),
                "decision": decision.to_dict(),
            },
        )
        write_json(
            self.output_dir / "evolution_controller_history.json",
            {
                "schema_version": 5,
                "max_physical_iterations": self.max_physical_iterations,
                "strategy_failure_budget": self.strategy_failure_budget,
                "physical_iterations": self.physical_iterations,
                "strategy_failure_counts": self.strategy_failure_counts,
                "exhausted_strategy_signatures": list(
                    self.exhausted_strategy_signatures
                ),
                "exhausted_motion_objectives": list(
                    self.exhausted_motion_objectives
                ),
                "attempts": [asdict(item) for item in self._attempts],
                "metrics": self.metrics(),
            },
        )

    def _load_history(self) -> None:
        directories = [*self.history_dirs]
        if self.output_dir is not None:
            directories.append(self.output_dir)
        if not directories:
            return
        attempts: list[object] = []
        seen_directories: set[Path] = set()
        for directory in directories:
            resolved = directory.resolve()
            if resolved in seen_directories:
                continue
            seen_directories.add(resolved)
            path = resolved / "evolution_controller_history.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidate_attempts = (
                payload.get("attempts", []) if isinstance(payload, dict) else []
            )
            if isinstance(candidate_attempts, list):
                attempts.extend(candidate_attempts)
        loaded: list[EvolutionAttempt] = []
        valid_actions = {
            "task_complete",
            "switch_strategy",
            "change_coordination",
            "rewrite_script",
            "tune_parameter",
            "continue",
            "repair_system",
            "stop_budget_exhausted",
        }
        for item in attempts:
            if not isinstance(item, dict):
                continue
            action = str(item.get("controller_action", "continue"))
            if action not in valid_actions:
                action = "continue"
            try:
                loaded.append(
                    EvolutionAttempt(
                        record_index=int(item.get("record_index", len(loaded) + 1)),
                        policy_attempt=int(item.get("policy_attempt", 0)),
                        script_id=str(item.get("script_id", "")),
                        checkpoint_id=str(item.get("checkpoint_id", "")),
                        strategy_signature=str(item.get("strategy_signature", "")),
                        motion_objective_signature=str(
                            item.get("motion_objective_signature", "")
                            or _motion_objective_from_parameter_signature(
                                str(item.get("strategy_signature", "")),
                                str(item.get("parameter_signature", "")),
                            )
                        ),
                        role_signature=str(item.get("role_signature", "")),
                        parameter_signature=str(item.get("parameter_signature", "")),
                        step_accepted=bool(item.get("step_accepted", False)),
                        task_complete=bool(item.get("task_complete", False)),
                        failure_type=str(item.get("failure_type", "none")),
                        failure_level=str(item.get("failure_level", "system")),
                        transition_outcome=str(
                            item.get(
                                "transition_outcome",
                                TransitionOutcome.ACCEPT_PROGRESS.value
                                if item.get("step_accepted", False)
                                else TransitionOutcome.SYSTEM_INVALID.value
                                if item.get("failure_level") == "system"
                                else TransitionOutcome.REJECT_UNSAFE.value,
                            )
                        ),
                        physically_dispatched=bool(item.get("physically_dispatched", False)),
                        counts_toward_budget=bool(item.get("counts_toward_budget", False)),
                        physical_iteration=int(item.get("physical_iteration", 0)),
                        strategy_failure_count=int(item.get("strategy_failure_count", 0)),
                        semantic_failure_count=int(item.get("semantic_failure_count", 0)),
                        route_stall_iterations=int(
                            item.get("route_stall_iterations", 0)
                        ),
                        strategy_failure_budget=int(
                            item.get("strategy_failure_budget", self.strategy_failure_budget)
                        ),
                        strategy_evidence_active=bool(
                            item.get("strategy_evidence_active", True)
                        ),
                        controller_action=action,  # type: ignore[arg-type]
                        should_evolve=bool(item.get("should_evolve", True)),
                        budget_remaining=int(item.get("budget_remaining", 0)),
                        hard_iteration_limit_reached=bool(
                            item.get("hard_iteration_limit_reached", False)
                        ),
                    )
                )
            except (TypeError, ValueError):
                continue
        deduplicated: dict[str, EvolutionAttempt] = {}
        for item in loaded:
            deduplicated[item.evidence_key] = item
        self._attempts = list(deduplicated.values())


_STRATEGY_MOVER_METHODS = frozenset(
    {
        "carry_forward",
        "carry_lateral_left",
        "carry_lateral_right",
        "turn_clockwise",
        "turn_counterclockwise",
    }
) | COORDINATION_PRIMITIVE_METHODS

_MOTION_OBJECTIVE_ARGUMENT_BY_METHOD = {
    "coordinated_linear_transport": "direction",
    "coordinated_arc_transport": "turn_direction",
    "coordinated_pivot": "turn_direction",
    "coordinated_alignment": "axis",
}

_MOTION_OBJECTIVE_CHOICES_BY_METHOD = {
    "coordinated_linear_transport": LOADED_TRANSLATION_DIRECTIONS,
    "coordinated_arc_transport": ("clockwise", "counterclockwise"),
    "coordinated_pivot": ("clockwise", "counterclockwise"),
    "coordinated_alignment": ("longitudinal", "lateral", "yaw"),
}


def _canonical_failure_class(failure_type: str) -> str:
    failure = str(failure_type).strip()
    if failure in {
        "no_payload_progress",
        "goal_regression",
        "route_regression",
        "straightness_regression",
        "alignment_regression",
    }:
        return "task_progress_regression"
    if failure in {
        "payload_drop",
        "payload_not_supported",
        "payload_slip",
        "payload_unstable",
    }:
        return "payload_support_failure"
    if failure in {"formation_desync", "robot_fall"}:
        return "team_coordination_failure"
    return failure or "unknown"


def _strategy_budget_key(strategy: str, objective: str) -> str:
    return f"{strategy}|{objective}" if objective else strategy


def _motion_objective_from_parameter_signature(
    strategy_signature: str,
    parameter_signature: str,
) -> str:
    try:
        method = strategy_signature.split(":", 1)[1]
        argument_name = _MOTION_OBJECTIVE_ARGUMENT_BY_METHOD[method]
        payload = json.loads(parameter_signature)
    except (IndexError, KeyError, json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(payload, list):
        return ""
    objectives: set[str] = set()
    for robot in payload:
        if not isinstance(robot, dict):
            return ""
        calls = robot.get("calls", [])
        if not isinstance(calls, list) or len(calls) != 1:
            return ""
        call = calls[0]
        arguments = call.get("arguments", {}) if isinstance(call, dict) else {}
        if not isinstance(arguments, dict):
            return ""
        objective = str(arguments.get(argument_name, ""))
        if not objective:
            return ""
        objectives.add(objective)
    return f"{method}:{next(iter(objectives))}" if len(objectives) == 1 else ""


def _strategy_has_coordination_repair(strategy_signature: str) -> bool:
    try:
        coordination_mode, method_name = strategy_signature.split(":", 1)
    except ValueError:
        return True
    return coordination_relationship_can_change(method_name, coordination_mode)


_ROLE_PARAMETER_BY_METHOD = {
    "coordinated_arc_transport": "radius_m",
    "coordinated_pivot": "role",
    "coordinated_alignment": "delta",
}


def _coordination_roles_exhausted(
    attempts: tuple[EvolutionAttempt, ...],
    current: EvolutionAttempt,
) -> bool:
    try:
        _coordination_mode, method_name = current.strategy_signature.split(":", 1)
    except ValueError:
        return False
    role_limit = coordination_role_variant_limit(method_name)
    if role_limit is None or role_limit <= 0:
        return False
    current_objective = _coordination_objective_signature(current)
    physical = [
        item
        for item in attempts
        if item.counts_toward_budget
        and item.strategy_evidence_active
        and item.checkpoint_id == current.checkpoint_id
    ]
    last_success_index = max(
        (
            index
            for index, item in enumerate(physical)
            if item.transition_outcome
            in {
                TransitionOutcome.ACCEPT_PROGRESS.value,
                TransitionOutcome.TASK_COMPLETE.value,
            }
            and item.strategy_signature == current.strategy_signature
        ),
        default=-1,
    )
    role_signatures = {
        item.role_signature
        for item in physical[last_success_index + 1 :]
        if item.strategy_signature == current.strategy_signature
        and not item.step_accepted
        and item.failure_level != "system"
        and item.role_signature
        and (
            not current_objective
            or not (objective := _coordination_objective_signature(item))
            or objective == current_objective
        )
    }
    return len(role_signatures) >= role_limit


def _coordination_objective_signature(attempt: EvolutionAttempt) -> str:
    try:
        method_name = attempt.strategy_signature.split(":", 1)[1]
        payload = json.loads(attempt.parameter_signature)
    except (IndexError, json.JSONDecodeError, TypeError):
        return ""
    role_parameter = _ROLE_PARAMETER_BY_METHOD.get(method_name)
    if role_parameter is None or not isinstance(payload, list):
        return ""
    shared_calls: set[str] = set()
    for robot in payload:
        if not isinstance(robot, dict):
            return ""
        calls = robot.get("calls", [])
        if not isinstance(calls, list) or len(calls) != 1:
            return ""
        call = calls[0]
        if not isinstance(call, dict):
            return ""
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            return ""
        shared_arguments = dict(arguments)
        shared_arguments.pop(role_parameter, None)
        shared_calls.add(
            json.dumps(
                {
                    "method": call.get("method", ""),
                    "arguments": shared_arguments,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    return next(iter(shared_calls)) if len(shared_calls) == 1 else ""


def policy_strategy_signature(policy: PolicyScript) -> str:
    """Return the exploration identity: coordination mode plus mover family."""

    scripts = _policy_scripts(policy)
    modes = {str(script.coordination_mode).strip().lower() or "sync" for script in scripts}
    methods = {
        method
        for script in scripts
        for method, _arguments in _task_calls(script.code)
        if method in _STRATEGY_MOVER_METHODS
    }
    if len(modes) != 1 or len(methods) != 1:
        return ""
    return f"{next(iter(modes))}:{next(iter(methods))}"


def policy_motion_objective_signature(policy: PolicyScript) -> str:
    """Return the shared semantic direction/axis used for physical budgeting."""

    strategy = policy_strategy_signature(policy)
    if not strategy:
        return ""
    method = strategy.split(":", 1)[-1]
    argument_name = _MOTION_OBJECTIVE_ARGUMENT_BY_METHOD.get(method)
    if argument_name is None:
        return ""
    objectives: set[str] = set()
    for script in _policy_scripts(policy):
        calls = _task_calls(script.code)
        if len(calls) != 1 or calls[0][0] != method:
            return ""
        objective = str(calls[0][1].get(argument_name, ""))
        if not objective:
            return ""
        objectives.add(objective)
    return f"{method}:{next(iter(objectives))}" if len(objectives) == 1 else ""


def _execution_repair_required_change(
    current: EvolutionAttempt,
    *,
    fallback: str,
) -> str:
    method = current.strategy_signature.rsplit(":", 1)[-1]
    unsafe_failures = {
        "robot_fall",
        "payload_drop",
        "payload_not_supported",
        "payload_unstable",
        "payload_slip",
        "formation_desync",
        "payload_wall_contact",
    }
    if (
        method
        not in {
            "coordinated_arc_transport",
            "coordinated_pivot",
        }
        or current.failure_type not in unsafe_failures
    ):
        if (
            method == "coordinated_linear_transport"
            and current.failure_type in unsafe_failures
        ):
            parameters = _uniform_transport_parameters(current)
            previous_values = (
                f" Current rejected values are distance_m={parameters[0]:g} and "
                f"speed_mps={parameters[1]:g}."
                if parameters is not None
                else ""
            )
            return (
                "reduce the loaded linear segment by exactly one supported distance tier; "
                "keep speed_mps=0.50, ramp_s=1.5, direction, mode, and the front/back "
                "relation fixed. Do not use speed reduction as a model-selected repair."
                + previous_values
            )
        return fallback
    parameters = _shared_turn_parameters(current)
    previous_values = (
        f" Current rejected shared values are angle_deg={parameters[0]:g} and "
        f"duration_s={parameters[1]:g}. Both generated robot code strings must "
        "literally change the selected numeric field; changing only rationale or plan text "
        "is invalid."
        if parameters is not None
        else ""
    )
    return (
        "keep turn direction, role assignment, and radii fixed; reduce angle_deg to no more "
        "than half of angle_deg_previous with duration_s unchanged. At the minimum angle, "
        "change coordination instead of extending duration_s."
        + previous_values
    )


def _linear_parameter_envelope_exhausted(
    current: EvolutionAttempt,
) -> bool:
    """Return whether a rejected loaded-linear action has no smaller SDK tier."""

    if (
        current.strategy_signature.rsplit(":", 1)[-1]
        != "coordinated_linear_transport"
        or current.failure_type
        not in {
            "robot_fall",
            "payload_drop",
            "payload_not_supported",
            "payload_unstable",
            "payload_slip",
            "formation_desync",
            "payload_wall_contact",
        }
    ):
        return False
    parameters = _uniform_transport_parameters(current)
    if parameters is None:
        return False
    return (
        parameters[0]
        <= min(COORDINATED_LINEAR_DISTANCE_CHOICES_M) + 1.0e-9
    )


def _uniform_transport_parameters(
    current: EvolutionAttempt,
) -> tuple[float, float] | None:
    try:
        payload = json.loads(current.parameter_signature)
    except (json.JSONDecodeError, TypeError):
        return None
    values: set[tuple[float, float]] = set()
    if not isinstance(payload, list):
        return None
    for robot in payload:
        if not isinstance(robot, dict):
            return None
        calls = robot.get("calls", [])
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        call = calls[0]
        if not isinstance(call, dict) or call.get("method") != "coordinated_linear_transport":
            return None
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            return None
        try:
            values.add(
                (float(arguments["distance_m"]), float(arguments["speed_mps"]))
            )
        except (KeyError, TypeError, ValueError):
            return None
    return next(iter(values)) if len(values) == 1 else None


def _shared_turn_parameters(
    current: EvolutionAttempt,
) -> tuple[float, float] | None:
    """Read one shared arc/pivot angle and duration from an attempt signature."""

    try:
        payload = json.loads(current.parameter_signature)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, list):
        return None
    values: set[tuple[float, float]] = set()
    for robot in payload:
        if not isinstance(robot, dict):
            return None
        calls = robot.get("calls", [])
        if not isinstance(calls, list) or len(calls) != 1:
            return None
        call = calls[0]
        if not isinstance(call, dict) or call.get("method") not in {
            "coordinated_arc_transport",
            "coordinated_pivot",
        }:
            return None
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            return None
        try:
            values.add(
                (float(arguments["angle_deg"]), float(arguments["duration_s"]))
            )
        except (KeyError, TypeError, ValueError):
            return None
    return next(iter(values)) if len(values) == 1 else None


def policy_role_signature(policy: PolicyScript) -> str:
    """Return the executable role relationship without numerical parameters."""

    parsed = []
    for script in sorted(_policy_scripts(policy), key=lambda item: item.robot_id):
        calls = _task_calls(script.code)
        method, arguments = calls[0] if calls else ("", {})
        parsed.append((script, method, arguments))
    arc_radii = {
        script.robot_id: _numeric_argument(arguments, "radius_m")
        for script, method, arguments in parsed
        if method == "coordinated_arc_transport"
    }
    payload = []
    for script, method, arguments in parsed:
        profile_role = ""
        if method == "coordinated_arc_transport" and len(arc_radii) == 2:
            own = arc_radii.get(script.robot_id)
            peer = next(
                (value for robot_id, value in arc_radii.items() if robot_id != script.robot_id),
                None,
            )
            if own is not None and peer is not None:
                profile_role = "outer" if own > peer else "inner"
        elif method == "coordinated_pivot":
            profile_role = str(arguments.get("role", ""))
        elif method == "coordinated_alignment":
            delta = _numeric_argument(arguments, "delta")
            profile_role = (
                "reference"
                if delta == 0.0
                else "positive_adjuster"
                if delta is not None and delta > 0.0
                else "negative_adjuster"
            )
        payload.append(
            {
                "robot_id": script.robot_id,
                "coordination_mode": script.coordination_mode,
                "method": method,
                "profile_role": profile_role,
                "contact_intent": script.contact_intent,
            }
        )
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _numeric_argument(arguments: dict[str, object], name: str) -> float | None:
    try:
        return float(arguments[name])
    except (KeyError, TypeError, ValueError):
        return None


def policy_parameter_signature(policy: PolicyScript) -> str:
    """Return normalized task-call parameters for parameter-only retry metrics."""

    payload = []
    for script in sorted(_policy_scripts(policy), key=lambda item: item.robot_id):
        payload.append(
            {
                "robot_id": script.robot_id,
                "calls": [
                    {"method": method, "arguments": arguments}
                    for method, arguments in _task_calls(script.code)
                ],
            }
        )
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def trace_has_physical_dispatch(trace: ExecutionTrace | None) -> bool:
    """Distinguish real backend calls from dry-run/script-validation attempts."""

    if trace is None:
        return False
    return any(
        str(call.status).strip().lower()
        not in {"", "dry_run", "dry_run_validated", "planned"}
        for call in trace.skill_calls
    )


def trace_physical_transaction_completed(trace: ExecutionTrace | None) -> bool:
    """Return whether a dispatched team transaction reached a terminal result.

    A monitor-triggered physical abort is still a completed, informative
    transaction. Preflight/schema failures and post-dispatch infrastructure
    failures are not. The latter is recorded on the evaluation report because
    a trace alone cannot distinguish an evaluator/provider failure after a
    successful publish from an ordinary completed rollout.
    """

    if not trace_has_physical_dispatch(trace) or trace is None:
        return False
    statuses = [
        str(call.status).strip().lower()
        for call in trace.skill_calls
        if str(call.status).strip()
        and call.skill_name != "stop_all"
        and call.effective_parameters.get("physical_publish") is not False
    ]
    if not statuses or set(statuses) & {"dry_run", "dry_run_validated", "planned"}:
        return False
    return all(status == "success" for status in statuses)


def _policy_scripts(policy: PolicyScript) -> tuple[RobotPolicyScript, ...]:
    if policy.robot_scripts:
        return policy.robot_scripts
    return (
        RobotPolicyScript(
            robot_id="front",
            script_id=f"front_{policy.script_id}",
            code=policy.code,
            synchronization_group="legacy_team",
            coordination_mode="sync",
        ),
    )


def _task_calls(code: str) -> list[tuple[str, dict[str, object]]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    calls: list[tuple[str, dict[str, object]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "api":
            continue
        method = node.func.attr
        if method == "stop_all":
            continue
        arguments: dict[str, object] = {}
        for index, argument in enumerate(node.args):
            arguments[f"arg_{index}"] = _literal_or_source(argument)
        for keyword in node.keywords:
            if keyword.arg is not None:
                arguments[keyword.arg] = _literal_or_source(keyword.value)
        calls.append((method, arguments))
    return calls


def _literal_or_source(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


__all__ = [
    "ControllerAction",
    "EvolutionAttempt",
    "EvolutionController",
    "EvolutionDecision",
    "EvolutionHistoryView",
    "policy_parameter_signature",
    "policy_motion_objective_signature",
    "policy_role_signature",
    "policy_strategy_signature",
    "trace_has_physical_dispatch",
    "trace_physical_transaction_completed",
]
