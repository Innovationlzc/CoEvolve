"""Task-progress acceptance layered on top of physical rollout evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import hypot
from typing import Any

from .contracts import BoxCarryTask, EvaluationReport, ExecutionTrace
from .failure_taxonomy import annotate_evaluation_report, classify_report
from .route_spec import route_segment_state
from .simulation_diagnostics import (
    advance_trusted_route_prefix,
    trusted_route_phase_guard,
)


class TransitionOutcome(str, Enum):
    """Mutually exclusive outcomes for one attempted physical transition."""

    ACCEPT_SETUP = "ACCEPT_SETUP"
    ACCEPT_PROGRESS = "ACCEPT_PROGRESS"
    TASK_COMPLETE = "TASK_COMPLETE"
    REJECT_UNSAFE = "REJECT_UNSAFE"
    REJECT_REGRESSION = "REJECT_REGRESSION"
    SYSTEM_INVALID = "SYSTEM_INVALID"


@dataclass(frozen=True)
class TransitionDecision:
    outcome: TransitionOutcome
    reason: str
    evidence: dict[str, Any]

    @property
    def trusted_checkpoint_allowed(self) -> bool:
        return self.outcome in {
            TransitionOutcome.ACCEPT_SETUP,
            TransitionOutcome.ACCEPT_PROGRESS,
            TransitionOutcome.TASK_COMPLETE,
        }

    @property
    def positive_route_memory_allowed(self) -> bool:
        return self.outcome in {
            TransitionOutcome.ACCEPT_PROGRESS,
            TransitionOutcome.TASK_COMPLETE,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "outcome": self.outcome.value,
            "trusted_checkpoint_allowed": self.trusted_checkpoint_allowed,
            "positive_route_memory_allowed": self.positive_route_memory_allowed,
        }


_REGRESSION_FAILURES = frozenset(
    {
        "no_payload_progress",
        "goal_regression",
        "route_regression",
        "straightness_regression",
        "alignment_regression",
    }
)
_SYSTEM_FAILURES = frozenset(
    {
        "runtime_state_missing",
        "manual_review_required",
        "script_error",
    }
)
_LOCAL_CORRECTION_ACTIONS = frozenset(
    {
        "coordinated_pivot",
        "coordinated_arc_transport",
        "coordinated_alignment",
        "turn_clockwise",
        "turn_counterclockwise",
    }
)
_LINEAR_ACTIONS = frozenset(
    {
        "coordinated_linear_transport",
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
    }
)
_LIFECYCLE_ACTIONS = frozenset(
    {
        "place_payload_on_arms",
        "place_crouch",
        "release_payload",
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
        "recover_payload_inward",
        "carry_recover_payload_inward",
        "recover_stand",
    }
)

SIGNIFICANT_ROUTE_REGRESSION_M = 0.10


def transition_outcome_from_report(report: EvaluationReport) -> TransitionOutcome:
    """Return the authoritative six-state outcome stored on an evaluation.

    Older engineering artifacts predate this contract. The compatibility
    mapping below is intentionally conservative; every new physical rollout is
    expected to carry ``evidence.transition_decision``.
    """

    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    decision = evidence.get("transition_decision", {})
    decision = decision if isinstance(decision, dict) else {}
    raw = str(decision.get("outcome", "")).strip()
    # A later checkpoint/runtime gate may invalidate an endpoint that initially
    # passed route classification. Never let the embedded pre-roundtrip
    # ACCEPT_* label override the final report disposition.
    if report.success:
        try:
            return TransitionOutcome(raw)
        except ValueError:
            if bool(report.metrics.get("task_complete", 0.0)):
                return TransitionOutcome.TASK_COMPLETE
            return TransitionOutcome.ACCEPT_PROGRESS
    classification = classify_report(report)
    if classification.level == "system" or report.failure_type in _SYSTEM_FAILURES:
        return TransitionOutcome.SYSTEM_INVALID
    if report.failure_type in _REGRESSION_FAILURES:
        return TransitionOutcome.REJECT_REGRESSION
    return TransitionOutcome.REJECT_UNSAFE


def transition_is_route_progress(report: EvaluationReport) -> bool:
    """Return whether an accepted transition may advance route semantics."""

    return transition_outcome_from_report(report) in {
        TransitionOutcome.ACCEPT_PROGRESS,
        TransitionOutcome.TASK_COMPLETE,
    }


def verify_transition(
    *,
    task: BoxCarryTask,
    report: EvaluationReport,
    trace: ExecutionTrace,
    before_state: dict[str, Any] | None,
    after_state: dict[str, Any],
    accepted_manifest: dict[str, Any] | None,
) -> TransitionDecision:
    """Decide whether a physically evaluated endpoint may enter trusted lineage."""

    if not report.success:
        classification = classify_report(report)
        if classification.level == "system" or report.failure_type in _SYSTEM_FAILURES:
            outcome = TransitionOutcome.SYSTEM_INVALID
        elif report.failure_type in _REGRESSION_FAILURES:
            outcome = TransitionOutcome.REJECT_REGRESSION
        else:
            outcome = TransitionOutcome.REJECT_UNSAFE
        return TransitionDecision(
            outcome=outcome,
            reason=f"existing evaluator rejection: {report.failure_type}",
            evidence={"source": "physics_and_vlm_evaluator"},
        )

    if bool(report.metrics.get("task_complete", 0.0)):
        return TransitionDecision(
            outcome=TransitionOutcome.TASK_COMPLETE,
            reason="the authoritative evaluator confirmed the full terminal contract",
            evidence={"source": "terminal_task_contract", "task_complete": True},
        )

    manifest = accepted_manifest if isinstance(accepted_manifest, dict) else {}
    accepted_steps = manifest.get("steps", ())
    accepted_steps = accepted_steps if isinstance(accepted_steps, list) else []
    before = before_state if isinstance(before_state, dict) else {}
    phase_guard = trusted_route_phase_guard(
        task,
        before,
        accepted_steps=accepted_steps,
    )
    action = _last_action(trace)
    turn_clearance_setup = _turn_clearance_setup_contract(trace)
    route_clearance_setup = _route_clearance_setup_contract(trace)
    lifecycle_setup = _lifecycle_setup_contract(trace)
    model_owned_setup = _model_owned_setup_contract(trace)
    turn_reversal = _locked_turn_episode_reversal(
        before_state=before,
        trace=trace,
        action=action,
    )
    if turn_reversal:
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason=(
                "candidate reversed an unfinished verifier-accepted signed turn "
                "without a rejected-turn switch authorization"
            ),
            evidence={
                "source": "turn_episode_continuity",
                "action": action,
                **turn_reversal,
            },
        )
    explicit_recovery = _explicit_linear_recovery_contract(trace)
    route_recovery_active = bool(
        str(explicit_recovery.get("kind", "")).strip() == "route_recovery"
    )
    recovery_geometry = (
        _recovery_geometry_improved(
            report,
            trace=trace,
            before_state=before,
            after_state=after_state,
            recovery=explicit_recovery,
            expected_scope_checkpoint_id=_latest_trusted_checkpoint_id(manifest),
        )
        if route_recovery_active
        else {}
    )
    anchor_regression = _phase_anchor_regression(
        task=task,
        after_state=after_state,
        phase_guard=phase_guard,
    )
    if anchor_regression:
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason="candidate moved substantially behind a completed phase anchor",
            evidence={
                "source": "phase_anchor_retention",
                "action": action,
                **anchor_regression,
            },
        )

    blocked_space_regression = _corroborated_blocked_space_regression(
        report,
        action=action,
    )
    if blocked_space_regression:
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason=(
                "measured motion and the fresh global observation jointly show "
                "entry toward blocked space"
            ),
            evidence={
                "source": "corroborated_global_route_evidence",
                "action": action,
                **blocked_space_regression,
            },
        )

    phase_progress = _active_phase_progress(
        task=task,
        before_state=before,
        after_state=after_state,
        phase_guard=phase_guard,
    )
    completed_prefix_before = int(phase_guard.get("maximum_completed_prefix", 0))
    before_segment = route_segment_state(
        task,
        before,
        completed_prefix=completed_prefix_before,
    )
    after_segment = route_segment_state(
        task,
        after_state,
        completed_prefix=completed_prefix_before,
    )
    topology_after = trusted_route_phase_guard(
        task,
        after_state,
        accepted_steps=accepted_steps,
        include_candidate_state=True,
    )
    raw_completed_prefix_after = int(
        topology_after.get("maximum_completed_prefix", completed_prefix_before)
    )
    active_gate_crossed = bool(phase_progress.get("active_gate_crossed", False))
    completed_prefix_after = (
        min(raw_completed_prefix_after, completed_prefix_before + 1)
        if active_gate_crossed
        else completed_prefix_before
    )
    topology_advanced = completed_prefix_after > completed_prefix_before
    corridor_recovery = _corridor_recovery(
        before_segment=before_segment,
        after_segment=after_segment,
    )
    local_setup = _local_setup_improved(report, action=action)
    model_owned_setup_effect = _model_owned_setup_effect(
        report,
        trace=trace,
        setup=model_owned_setup,
        local_setup=local_setup,
        expected_scope_checkpoint_id=_latest_trusted_checkpoint_id(manifest),
    )
    local_correction_improved = bool(
        action in _LOCAL_CORRECTION_ACTIONS
        and local_setup.get("improved", False)
    )
    # A pivot/alignment is judged by its declared local relation. Small endpoint
    # translation during a physically safe, correct correction must not override
    # measured heading/formation improvement. Wall contact, blocked-space
    # corroboration, phase-anchor reversal, and evaluator safety failures have
    # already been checked above.
    corridor_regression = (
        {}
        if local_correction_improved
        else _corridor_regression(
            action=action,
            before_segment=before_segment,
            after_segment=after_segment,
            topology_advanced=topology_advanced,
        )
    )
    if corridor_regression:
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason="candidate left the active connected-route corridor envelope",
            evidence={
                "source": "route_corridor_envelope",
                "action": action,
                **corridor_regression,
            },
        )

    route_regression = _significant_active_route_regression(
        action=action,
        phase_progress=phase_progress,
        topology_advanced=topology_advanced,
        corridor_recovery=corridor_recovery,
        trace=trace,
    )
    if (
        route_regression
        and not turn_clearance_setup
        and not model_owned_setup_effect.get("improved", False)
        and not (
            route_recovery_active and recovery_geometry.get("improved", False)
        )
    ):
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason=(
                "ordinary linear transport moved substantially opposite the "
                "active connected-route progress coordinate"
            ),
            evidence={
                "source": "active_route_progress_regression",
                "action": action,
                "explicit_recovery_contract": explicit_recovery,
                "recovery_geometry": recovery_geometry,
                **route_regression,
            },
        )

    straightness_regression = _severe_straightness_regression(
        report,
        action=action,
    )
    if straightness_regression:
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason="linear transport exceeded the generic severe-deviation boundary",
            evidence={
                "source": "linear_straightness_contract",
                "action": action,
                **straightness_regression,
            },
        )

    cycle = detect_state_progress_cycle(
        task=task,
        accepted_manifest=manifest,
        candidate_report=report,
        candidate_state=after_state,
        candidate_phase_prefix=(
            completed_prefix_after
            if action in _LINEAR_ACTIONS
            and not route_recovery_active
            and not turn_clearance_setup
            and not route_clearance_setup
            and not lifecycle_setup
            and not model_owned_setup
            else completed_prefix_before
        ),
    )
    if cycle.get("detected", False) and not local_correction_improved:
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason=str(cycle.get("reason", "state/progress cycle detected")),
            evidence={
                "source": "state_progress_cycle_detector",
                "action": action,
                **cycle,
            },
        )

    if action in _LOCAL_CORRECTION_ACTIONS and not local_setup.get(
        "improved", False
    ):
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason="local correction was safe but did not improve its declared relation",
            evidence={
                "source": "local_setup_effect",
                "action": action,
                **local_setup,
            },
        )
    no_effect = _declared_action_no_effect(report, action=action)
    if no_effect:
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason="the executable motion transaction had no measurable declared effect",
            evidence={
                "source": "declared_action_effect",
                "action": action,
                **no_effect,
            },
        )
    if route_recovery_active and not recovery_geometry.get("improved", False):
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason=(
                "checkpoint-scoped route recovery did not produce a signed "
                "clearance or commanded-direction improvement"
            ),
            evidence={
                "source": "explicit_route_recovery_effect",
                "action": action,
                "explicit_recovery_contract": explicit_recovery,
                "recovery_geometry": recovery_geometry,
            },
        )
    if model_owned_setup and not model_owned_setup_effect.get("improved", False):
        return TransitionDecision(
            outcome=TransitionOutcome.REJECT_REGRESSION,
            reason=(
                "model-declared setup did not produce a direction-consistent "
                "measurable setup effect"
            ),
            evidence={
                "source": "model_owned_setup_effect",
                "action": action,
                "model_owned_setup": model_owned_setup,
                "model_owned_setup_effect": model_owned_setup_effect,
            },
        )
    route_progress = _route_progress_evidence(
        action=action,
        before_segment=before_segment,
        after_segment=after_segment,
        topology_advanced=topology_advanced,
        completed_prefix_before=completed_prefix_before,
        completed_prefix_after=completed_prefix_after,
        report=report,
    )
    # Returning to the authored corridor is setup evidence unless that same
    # physical transaction crosses an oriented topology gate.  A gate crossing
    # is authoritative progress and must not be hidden by the recovery label.
    route_progress_accepted = bool(
        route_progress.get("progress", False)
        and not route_recovery_active
        and (
            route_progress.get("topology_advanced", False)
            or not corridor_recovery.get("active", False)
        )
    )
    outcome = (
        TransitionOutcome.ACCEPT_PROGRESS
        if route_progress_accepted
        and not turn_clearance_setup
        and not route_clearance_setup
        and not lifecycle_setup
        and not model_owned_setup
        else TransitionOutcome.ACCEPT_SETUP
    )
    return TransitionDecision(
        outcome=outcome,
        reason=(
            "wall-clear translation created room for the preserved signed turn"
            if turn_clearance_setup
            else (
                "route-clearance maneuver created setup room while preserving "
                "the route reference"
            )
            if route_clearance_setup
            else "placement-readiness maneuver completed as lifecycle setup"
            if lifecycle_setup
            else "connected-route topology or local-goal progress was measured"
            if outcome is TransitionOutcome.ACCEPT_PROGRESS
            else "the endpoint is physically restorable setup without route progress"
        ),
        evidence={
            "source": "transition_contract",
            "action": action,
            "phase_guard_before": phase_guard,
            "phase_guard_after": topology_after,
            "active_phase_progress": phase_progress,
            "route_segment_before": before_segment,
            "route_segment_after": after_segment,
            "route_progress": route_progress,
            "raw_completed_prefix_after": raw_completed_prefix_after,
            "active_gate_crossed": active_gate_crossed,
            "corridor_recovery": corridor_recovery,
            "explicit_recovery_contract": explicit_recovery,
            "recovery_geometry": recovery_geometry,
            "turn_clearance_setup": turn_clearance_setup,
            "route_clearance_setup": route_clearance_setup,
            "lifecycle_setup": lifecycle_setup,
            "model_owned_setup": model_owned_setup,
            "model_owned_setup_effect": model_owned_setup_effect,
            "local_setup": local_setup,
            "local_correction_envelope_applies": action in _LOCAL_CORRECTION_ACTIONS,
            "corridor_position_gate_deferred_to_improving_local_correction": (
                local_correction_improved
            ),
        },
    )


def apply_transition_decision(
    report: EvaluationReport,
    decision: TransitionDecision,
) -> EvaluationReport:
    """Attach the transition decision and turn regression into evaluator failure."""

    evidence = {**report.evidence, "transition_decision": decision.to_dict()}
    if decision.trusted_checkpoint_allowed:
        return annotate_evaluation_report(replace(report, evidence=evidence))
    if not report.success:
        return annotate_evaluation_report(replace(report, evidence=evidence))
    failure_type = (
        "route_regression"
        if decision.outcome is TransitionOutcome.REJECT_REGRESSION
        else "manual_review_required"
        if decision.outcome is TransitionOutcome.SYSTEM_INVALID
        else "payload_unstable"
    )
    return annotate_evaluation_report(
        replace(
            report,
            success=False,
            failure_type=failure_type,
            failed_subtask=str(decision.evidence.get("action", report.failed_subtask)),
            evidence=evidence,
            recommendations=tuple(report.recommendations)
            + (
                "restore the latest trusted checkpoint; do not append this endpoint to trusted lineage",
            ),
        )
    )


def detect_state_progress_cycle(
    *,
    task: BoxCarryTask,
    accepted_manifest: dict[str, Any],
    candidate_report: EvaluationReport,
    candidate_state: dict[str, Any],
    candidate_phase_prefix: int | None = None,
    recent_window: int = 8,
) -> dict[str, Any]:
    """Detect geometric recurrence and high-path/low-net oscillation."""

    states = _manifest_route_states(task, accepted_manifest)
    candidate = _candidate_route_state(
        candidate_report,
        candidate_state,
        states,
        phase_prefix=candidate_phase_prefix,
    )
    if candidate is None:
        return {"detected": False, "reason": "candidate pose unavailable"}
    states.append(candidate)
    recent = states[-max(3, int(recent_window)) :]
    signatures = [_coarse_state_signature(task, item) for item in recent]
    current_signature = signatures[-1]
    prior_matches = [
        index
        for index, signature in enumerate(signatures[:-1])
        if signature == current_signature and len(signatures) - 1 - index >= 3
        and any(
            intermediate != current_signature
            for intermediate in signatures[index + 1 : -1]
        )
    ]
    phase_advanced = max(signature[0] for signature in signatures) > signatures[0][0]
    if prior_matches and not phase_advanced:
        return {
            "detected": True,
            "kind": "coarse_state_recurrence",
            "reason": "recent trusted/candidate trajectory returned to a prior coarse state without phase advancement",
            "recent_state_count": len(recent),
            "matching_recent_indices": prior_matches,
            "current_signature": list(current_signature),
            "departed_before_return": True,
            "cycle_semantics_version": 2,
        }

    path_m = sum(
        hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        for previous, current in zip(recent, recent[1:])
    )
    net_m = hypot(
        recent[-1]["x"] - recent[0]["x"],
        recent[-1]["y"] - recent[0]["y"],
    )
    if path_m > 2.0 and net_m < 0.25 and not phase_advanced:
        return {
            "detected": True,
            "kind": "high_path_low_net_progress",
            "reason": "recent payload path exceeded 2m while net displacement stayed below 0.25m without phase advancement",
            "recent_state_count": len(recent),
            "cumulative_path_m": round(path_m, 6),
            "net_displacement_m": round(net_m, 6),
        }
    return {
        "detected": False,
        "recent_state_count": len(recent),
        "cumulative_path_m": round(path_m, 6),
        "net_displacement_m": round(net_m, 6),
    }


def _corridor_regression(
    *,
    action: str,
    before_segment: dict[str, Any],
    after_segment: dict[str, Any],
    topology_advanced: bool,
) -> dict[str, Any]:
    """Reject leaving authored free space, not temporary along-route retreat."""

    if topology_advanced or action not in (_LINEAR_ACTIONS | _LOCAL_CORRECTION_ACTIONS):
        return {}
    if not before_segment.get("available", False) or not after_segment.get(
        "available", False
    ):
        return {}
    before_inside = bool(before_segment.get("inside_corridor_envelope", False))
    after_inside = bool(after_segment.get("inside_corridor_envelope", False))
    if after_inside:
        return {}
    before_violation = _number(
        before_segment.get("corridor_envelope_violation_m")
    )
    after_violation = _number(
        after_segment.get("corridor_envelope_violation_m")
    )
    recovery_improvement_m = before_violation - after_violation
    if not before_inside and recovery_improvement_m >= 0.01:
        return {}
    return {
        "violated": True,
        "segment_id": str(after_segment.get("segment_id", "")),
        "before_inside_corridor_envelope": before_inside,
        "after_inside_corridor_envelope": after_inside,
        "before_corridor_envelope_violation_m": before_violation,
        "after_corridor_envelope_violation_m": after_violation,
        "corridor_recovery_improvement_m": round(recovery_improvement_m, 6),
        "before_cross_track_error_m": before_segment.get("cross_track_error_m"),
        "after_cross_track_error_m": after_segment.get("cross_track_error_m"),
        "corridor_half_width_m": after_segment.get("corridor_half_width_m"),
        "detour_margin_m": after_segment.get("detour_margin_m"),
        "temporary_along_route_retreat_is_allowed": True,
        "outside_to_inside_or_smaller_violation_is_allowed": True,
    }


def _corridor_recovery(
    *,
    before_segment: dict[str, Any],
    after_segment: dict[str, Any],
) -> dict[str, Any]:
    """Identify a safe outside-to-inside recovery without calling it route progress."""

    if not before_segment.get("available", False) or not after_segment.get(
        "available", False
    ):
        return {"active": False}
    if bool(before_segment.get("inside_corridor_envelope", False)):
        return {"active": False}
    before_violation = _number(
        before_segment.get("corridor_envelope_violation_m")
    )
    after_violation = _number(
        after_segment.get("corridor_envelope_violation_m")
    )
    improvement = before_violation - after_violation
    active = bool(
        after_segment.get("inside_corridor_envelope", False)
        or improvement >= 0.01
    )
    return {
        "active": active,
        "before_violation_m": round(before_violation, 6),
        "after_violation_m": round(after_violation, 6),
        "improvement_m": round(improvement, 6),
        "entered_corridor": bool(
            after_segment.get("inside_corridor_envelope", False)
        ),
    }


def _significant_active_route_regression(
    *,
    action: str,
    phase_progress: dict[str, Any],
    topology_advanced: bool,
    corridor_recovery: dict[str, Any],
    trace: ExecutionTrace,
) -> dict[str, Any]:
    """Reject large route reversal while preserving explicit recovery motion."""

    if action not in _LINEAR_ACTIONS or topology_advanced:
        return {}
    if not phase_progress.get("available", False):
        return {}
    if corridor_recovery.get("active", False):
        return {}
    delta_m = _number(phase_progress.get("delta_m"))
    if delta_m > -SIGNIFICANT_ROUTE_REGRESSION_M:
        return {}
    return {
        "violated": True,
        "progress_coordinate": str(
            phase_progress.get("progress_coordinate", "")
        ),
        "route_progress_before": phase_progress.get("before"),
        "route_progress_after": phase_progress.get("after"),
        "route_progress_delta_m": round(delta_m, 6),
        "maximum_uncontracted_retreat_m": SIGNIFICANT_ROUTE_REGRESSION_M,
        "explicit_recovery_contract": False,
    }


_MIN_RECOVERY_DISPLACEMENT_M = 0.05


def _recovery_geometry_improved(
    report: EvaluationReport,
    *,
    trace: ExecutionTrace,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    recovery: dict[str, Any],
    expected_scope_checkpoint_id: str,
) -> dict[str, Any]:
    """Measure whether an explicit recovery transaction improved setup geometry.

    Prefer existing wall/corridor clearance deltas when present. Otherwise use
    the motor-side endpoint projection along the already selected recovery
    direction.  Euclidean displacement alone is not evidence: it cannot tell a
    useful escape from lateral drift or motion in the opposite direction.
    """

    direction_evidence = _recovery_direction_evidence(trace, recovery)
    direction = str(direction_evidence.get("declared_direction", ""))
    source = str(recovery.get("source", "")).strip()
    scope_checkpoint_id = str(recovery.get("scope_checkpoint_id", "")).strip()
    contract_authoritative = bool(
        scope_checkpoint_id
        and expected_scope_checkpoint_id
        and scope_checkpoint_id == expected_scope_checkpoint_id
        and source
        and source != "committed_subtask_setup_only"
        and recovery.get("host_authoritative") is True
        and direction_evidence.get("consistent") is True
    )
    metrics = report.metrics if isinstance(report.metrics, dict) else {}
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    wall_before = _number(
        metrics.get("recovery_wall_clearance_before_m"),
        default=float("nan"),
    )
    wall_after = _number(
        metrics.get("recovery_wall_clearance_after_m"),
        default=float("nan"),
    )
    if wall_before != wall_before:  # NaN
        wall_before = _number(
            (evidence.get("recovery_wall_clearance") or {}).get("before_m")
            if isinstance(evidence.get("recovery_wall_clearance"), dict)
            else None,
            default=float("nan"),
        )
        wall_after = _number(
            (evidence.get("recovery_wall_clearance") or {}).get("after_m")
            if isinstance(evidence.get("recovery_wall_clearance"), dict)
            else None,
            default=float("nan"),
        )
    wall_delta = (
        wall_after - wall_before
        if wall_before == wall_before and wall_after == wall_after
        else None
    )
    signed_progress_m = _recovery_signed_command_progress_m(trace)
    wall_improved = bool(wall_delta is not None and wall_delta >= 0.01)
    signed_progress_improved = bool(
        signed_progress_m is not None
        and signed_progress_m >= _MIN_RECOVERY_DISPLACEMENT_M
    )
    if not contract_authoritative:
        improved = False
        metric = "invalid_recovery_contract"
    elif wall_improved and signed_progress_improved:
        improved = True
        metric = "wall_clearance_and_signed_command_progress"
    elif wall_improved:
        improved = True
        metric = "wall_clearance"
    elif signed_progress_improved:
        improved = True
        metric = "signed_command_progress"
    else:
        improved = False
        metric = "no_verified_recovery_improvement"
    return {
        "improved": bool(improved),
        "metric": metric,
        "contract_authoritative": contract_authoritative,
        "source": source,
        "scope_checkpoint_id": scope_checkpoint_id,
        "expected_scope_checkpoint_id": expected_scope_checkpoint_id,
        "command_direction": direction,
        "direction_evidence": direction_evidence,
        "wall_clearance_delta_m": (
            None if wall_delta is None else round(wall_delta, 6)
        ),
        "wall_clearance_improved": wall_improved,
        "signed_command_progress_m": (
            None if signed_progress_m is None else round(signed_progress_m, 6)
        ),
        "signed_command_progress_improved": signed_progress_improved,
        "minimum_recovery_displacement_m": _MIN_RECOVERY_DISPLACEMENT_M,
    }


def _latest_trusted_checkpoint_id(manifest: dict[str, Any]) -> str:
    """Return the checkpoint that a one-shot recovery contract may bind to."""

    latest = str(manifest.get("latest_checkpoint", "")).strip()
    if latest:
        return latest
    steps = manifest.get("steps", ())
    if isinstance(steps, list):
        for step in reversed(steps):
            if not isinstance(step, dict):
                continue
            checkpoint_id = str(step.get("checkpoint_after", "")).strip()
            if checkpoint_id:
                return checkpoint_id
    return str(manifest.get("initial_checkpoint", "")).strip()


def _recovery_signed_command_progress_m(trace: ExecutionTrace) -> float | None:
    """Read the signed endpoint measurement produced by the motor controller."""

    for call in reversed(trace.skill_calls):
        payload = (
            call.effective_parameters
            if isinstance(call.effective_parameters, dict)
            and call.effective_parameters
            else call.parameters
        )
        payload = payload if isinstance(payload, dict) else {}
        team_result = payload.get("coordinated_team_result", {})
        team_result = team_result if isinstance(team_result, dict) else {}
        feedback = team_result.get("linear_distance_feedback", {})
        feedback = feedback if isinstance(feedback, dict) else {}
        try:
            value = float(feedback["settled_payload_progress_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if value == value:  # exclude NaN without adding another numeric dependency
            return value
    return None


def _recovery_direction_evidence(
    trace: ExecutionTrace,
    recovery: dict[str, Any],
) -> dict[str, Any]:
    """Verify one Host-bound recovery direction against the executable trace."""

    declared = str(recovery.get("semantic_objective", "")).strip().lower()
    executed_directions: set[str] = set()
    feedback_directions: set[str] = set()
    executable_methods: set[str] = set()
    for call in trace.skill_calls:
        if call.skill_name == "stop_all":
            continue
        parameters = call.parameters if isinstance(call.parameters, dict) else {}
        direct_direction = str(parameters.get("direction", "")).strip().lower()
        if direct_direction:
            executed_directions.add(direct_direction)
        if call.skill_name:
            executable_methods.add(str(call.skill_name).strip())
        actions = parameters.get("actions", {})
        actions = actions if isinstance(actions, dict) else {}
        for action in actions.values():
            action = action if isinstance(action, dict) else {}
            method = str(action.get("method", "")).strip()
            if method:
                executable_methods.add(method)
            nested = action.get("parameters", {})
            nested = nested if isinstance(nested, dict) else {}
            nested_direction = str(nested.get("direction", "")).strip().lower()
            if nested_direction:
                executed_directions.add(nested_direction)
        effective = (
            call.effective_parameters
            if isinstance(call.effective_parameters, dict)
            else {}
        )
        team_result = effective.get("coordinated_team_result", {})
        team_result = team_result if isinstance(team_result, dict) else {}
        feedback = team_result.get("linear_distance_feedback", {})
        feedback = feedback if isinstance(feedback, dict) else {}
        feedback_direction = str(
            feedback.get("semantic_direction", "")
        ).strip().lower()
        if feedback_direction:
            feedback_directions.add(feedback_direction)
    allowed = {"forward", "backward", "left", "right"}
    consistent = bool(
        declared in allowed
        and executed_directions == {declared}
        and (not feedback_directions or feedback_directions == {declared})
        and executable_methods == {"coordinated_linear_transport"}
    )
    return {
        "consistent": consistent,
        "declared_direction": declared,
        "executed_directions": sorted(executed_directions),
        "feedback_directions": sorted(feedback_directions),
        "executable_methods": sorted(executable_methods),
    }


def _explicit_linear_recovery_contract(trace: ExecutionTrace) -> dict[str, Any]:
    contract = (
        trace.verification_contract
        if isinstance(trace.verification_contract, dict)
        else {}
    )
    kind = str(contract.get("kind", "")).strip()
    if kind == "formation_sync_repair_probe":
        return {"kind": kind}
    route_recovery = contract.get("route_recovery", {})
    route_recovery = (
        route_recovery if isinstance(route_recovery, dict) else {}
    )
    if route_recovery.get("active", False):
        return {"kind": "route_recovery", **route_recovery}
    turn_clearance = _turn_clearance_setup_contract(trace)
    if turn_clearance:
        return {"kind": "turn_clearance_replan", **turn_clearance}
    route_clearance = _route_clearance_setup_contract(trace)
    if route_clearance:
        return {"kind": "route_clearance_setup", **route_clearance}
    return {}


def _turn_clearance_setup_contract(trace: ExecutionTrace) -> dict[str, Any]:
    contract = (
        trace.verification_contract
        if isinstance(trace.verification_contract, dict)
        else {}
    )
    setup = contract.get("turn_clearance_replan", {})
    setup = setup if isinstance(setup, dict) else {}
    if not (
        setup.get("active", False)
        and setup.get("setup_only", False)
        and not setup.get("phase_completion_authorized", False)
    ):
        return {}
    return {
        "active": True,
        "scope_checkpoint_id": str(setup.get("scope_checkpoint_id", "")),
        "preserve_turn_direction": str(
            setup.get("preserve_turn_direction", "")
        ),
        "allowed_translation_directions": list(
            setup.get("allowed_translation_directions", ())
        ),
        "setup_only": True,
        "phase_completion_authorized": False,
    }


def _route_clearance_setup_contract(trace: ExecutionTrace) -> dict[str, Any]:
    """Return one model-selected maneuver bound to a route-clearance conflict."""

    contract = (
        trace.verification_contract
        if isinstance(trace.verification_contract, dict)
        else {}
    )
    setup = contract.get("route_clearance_conflict", {})
    setup = setup if isinstance(setup, dict) else {}
    committed = contract.get("committed_subtask", {})
    committed = committed if isinstance(committed, dict) else {}
    method = str(committed.get("primitive", "")).strip()
    objective = str(committed.get("semantic_objective", "")).strip()
    allowed_turns = {
        str(item).strip()
        for item in setup.get("safe_heading_correction_turn_directions", ())
        if str(item).strip()
    }
    allowed_translations = {
        str(item).strip()
        for key in (
            "route_reentry_translation_directions",
            "clearance_setup_translation_directions",
        )
        for item in setup.get(key, ())
        if str(item).strip()
    }
    selected_setup = bool(
        (method == "coordinated_pivot" and objective in allowed_turns)
        or (
            method == "coordinated_linear_transport"
            and objective in allowed_translations
        )
    )
    if not (
        setup.get("active", False)
        and setup.get("setup_only", False)
        and not setup.get("phase_completion_authorized", False)
        and selected_setup
        and _last_action(trace) == method
    ):
        return {}
    return {
        "active": True,
        "scope_checkpoint_id": str(setup.get("scope_checkpoint_id", "")),
        "primitive": method,
        "semantic_objective": objective,
        "preserve_navigation_world_direction_xy": list(
            setup.get("preserve_navigation_world_direction_xy", ())
        ),
        "setup_only": True,
        "phase_completion_authorized": False,
    }


def _model_owned_setup_contract(trace: ExecutionTrace) -> dict[str, Any]:
    """Return a route-neutral setup classification bound from the model plan.

    The Host does not choose this action. It only binds the already validated
    Task Orchestrator declaration to the exact executable transaction so the
    verifier cannot mistake temporary setup motion for route progress.
    """

    contract = (
        trace.verification_contract
        if isinstance(trace.verification_contract, dict)
        else {}
    )
    committed = contract.get("committed_subtask", {})
    committed = committed if isinstance(committed, dict) else {}
    method = str(committed.get("primitive", "")).strip()
    reference = str(committed.get("navigation_reference", "")).strip()
    allowed = _LINEAR_ACTIONS | _LOCAL_CORRECTION_ACTIONS
    if not (
        committed.get("source")
        == "host_binding_from_validated_task_orchestrator_plan"
        and committed.get("immutable_by_coder") is True
        and committed.get("setup_only") is True
        and committed.get("phase_completion_authorized") is False
        and committed.get("positive_route_memory_eligible") is False
        and method in allowed
        and _last_action(trace) == method
        and not (
            reference == "formation_only"
            and method in (_LINEAR_ACTIONS | (_LOCAL_CORRECTION_ACTIONS - {"coordinated_alignment"}))
        )
    ):
        return {}
    return {
        "active": True,
        "source": "validated_model_subtask_setup",
        "scope_checkpoint_id": str(committed.get("checkpoint_id", "")).strip(),
        "primitive": method,
        "semantic_objective": str(
            committed.get("semantic_objective", "")
        ).strip(),
        "navigation_reference": reference,
        "setup_only": True,
        "phase_completion_authorized": False,
        "positive_route_memory_eligible": False,
    }


def _model_owned_setup_effect(
    report: EvaluationReport,
    *,
    trace: ExecutionTrace,
    setup: dict[str, Any],
    local_setup: dict[str, Any],
    expected_scope_checkpoint_id: str,
) -> dict[str, Any]:
    """Require an exact scoped setup to have the model-declared physical effect."""

    if not setup:
        return {}
    scope = str(setup.get("scope_checkpoint_id", "")).strip()
    scope_matches = bool(
        scope
        and expected_scope_checkpoint_id
        and scope == expected_scope_checkpoint_id
    )
    method = str(setup.get("primitive", "")).strip()
    if method in _LINEAR_ACTIONS:
        direction = _recovery_direction_evidence(trace, setup)
        signed_progress_m = _recovery_signed_command_progress_m(trace)
        effect_matches = bool(
            direction.get("consistent") is True
            and signed_progress_m is not None
            and signed_progress_m >= _MIN_RECOVERY_DISPLACEMENT_M
        )
        return {
            "improved": bool(scope_matches and effect_matches),
            "metric": "signed_command_progress",
            "scope_matches": scope_matches,
            "direction_evidence": direction,
            "signed_command_progress_m": signed_progress_m,
            "minimum_displacement_m": _MIN_RECOVERY_DISPLACEMENT_M,
        }
    effect_matches = bool(local_setup.get("improved", False))
    return {
        "improved": bool(scope_matches and effect_matches),
        "metric": "declared_local_relation",
        "scope_matches": scope_matches,
        "local_setup": dict(local_setup),
    }


def _lifecycle_setup_contract(trace: ExecutionTrace) -> dict[str, Any]:
    """Return the Host marker for a pre-release readiness maneuver."""

    contract = (
        trace.verification_contract
        if isinstance(trace.verification_contract, dict)
        else {}
    )
    setup = contract.get("lifecycle_setup", {})
    setup = setup if isinstance(setup, dict) else {}
    if not (
        setup.get("active", False)
        and str(setup.get("lifecycle_state", "")).strip() in {"place", "release"}
        and setup.get("setup_only", False)
        and not setup.get("phase_completion_authorized", False)
        and not setup.get("positive_route_memory_eligible", False)
    ):
        return {}
    return {
        "active": True,
        "lifecycle_state": str(setup.get("lifecycle_state", "")).strip(),
        "objective": str(setup.get("objective", "")).strip(),
        "setup_only": True,
        "phase_completion_authorized": False,
        "positive_route_memory_eligible": False,
    }


def _severe_straightness_regression(
    report: EvaluationReport,
    *,
    action: str,
) -> dict[str, Any]:
    if action not in _LINEAR_ACTIONS:
        return {}
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    straight = evidence.get("straight_transport", {})
    straight = straight if isinstance(straight, dict) else {}
    formation = evidence.get("linear_formation_geometry", {})
    formation = formation if isinstance(formation, dict) else {}
    severe = bool(
        straight.get("severe_deviation", False)
        or formation.get("hard_failure", False)
    )
    if not severe:
        return {}
    return {
        "violated": True,
        "straight_transport": straight,
        "linear_formation_geometry": formation,
    }


def _corroborated_blocked_space_regression(
    report: EvaluationReport,
    *,
    action: str,
) -> dict[str, Any]:
    """Consume only route evidence already corroborated by measured execution."""

    if action not in (_LINEAR_ACTIONS | _LOCAL_CORRECTION_ACTIONS):
        return {}
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    route = evidence.get("route_direction_verification", {})
    route = route if isinstance(route, dict) else {}
    if (
        route.get("decision") != "reject_executed_route_direction"
        or route.get("risk_basis") != "executed_motion_into_blocked_space"
    ):
        return {}
    try:
        displacement_m = float(route.get("payload_step_displacement_xy_m", 0.0))
    except (TypeError, ValueError):
        displacement_m = 0.0
    corroboration = route.get("physics_corroboration", {})
    corroboration = corroboration if isinstance(corroboration, dict) else {}
    corroborated = bool(
        displacement_m >= 0.05
        and any(
            bool(corroboration.get(key, False))
            for key in (
                "wall_contact",
                "reduced_wall_clearance",
                "visible_near_wall_regression",
            )
        )
    )
    if not corroborated:
        return {}
    return {
        "violated": True,
        "risk_basis": route.get("risk_basis"),
        "route_direction_evidence": route.get("route_direction_evidence", ""),
        "payload_step_displacement_xy_m": displacement_m,
        "physics_corroboration": corroboration,
        "model_opinion_alone_is_advisory": True,
    }


def _declared_action_no_effect(
    report: EvaluationReport,
    *,
    action: str,
) -> dict[str, Any]:
    if action in _LIFECYCLE_ACTIONS:
        return {}
    if action in _LOCAL_CORRECTION_ACTIONS:
        # Local corrections are handled by their relation-specific improvement gate.
        return {}
    if action not in _LINEAR_ACTIONS and action not in {"", "stop_all"}:
        return {}
    displacement = _number(
        report.metrics.get(
            "payload_step_displacement_xy_m",
            report.metrics.get("payload_step_displacement_m", 0.0),
        )
    )
    required = max(
        0.005,
        _number(report.metrics.get("required_step_displacement_m", 0.0)),
    )
    if action in _LINEAR_ACTIONS and displacement >= required:
        return {}
    return {
        "violated": True,
        "payload_displacement_m": round(displacement, 6),
        "required_displacement_m": round(required, 6),
    }


def _local_setup_improved(
    report: EvaluationReport,
    *,
    action: str,
) -> dict[str, Any]:
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    local_navigation = evidence.get("local_navigation_transition", {})
    local_navigation = (
        local_navigation if isinstance(local_navigation, dict) else {}
    )
    alignment = evidence.get("alignment_progress_verification", {})
    alignment = alignment if isinstance(alignment, dict) else {}
    improved = bool(
        local_navigation.get("semantic_heading_improved", False)
        or alignment.get("semantic_alignment_improved", False)
    )
    return {
        "action": action,
        "improved": improved,
        "heading_improved": bool(
            local_navigation.get("semantic_heading_improved", False)
        ),
        "formation_improved": bool(
            alignment.get("semantic_alignment_improved", False)
        ),
    }


def _route_progress_evidence(
    *,
    action: str,
    before_segment: dict[str, Any],
    after_segment: dict[str, Any],
    topology_advanced: bool,
    completed_prefix_before: int,
    completed_prefix_after: int,
    report: EvaluationReport,
) -> dict[str, Any]:
    """Measure route progress independently of physical checkpoint acceptance."""

    local_goal_delta_m = 0.0
    along_delta_m = 0.0
    if before_segment.get("available", False) and after_segment.get(
        "available", False
    ):
        try:
            local_goal_delta_m = float(
                before_segment["local_goal_distance_m"]
            ) - float(after_segment["local_goal_distance_m"])
            along_delta_m = float(
                after_segment["along_segment_progress_m"]
            ) - float(before_segment["along_segment_progress_m"])
        except (KeyError, TypeError, ValueError):
            local_goal_delta_m = 0.0
            along_delta_m = 0.0
    # A correction can be essential and restorable, but it is not positive route
    # evidence until a subsequent transport moves through connected free space.
    # One scalar may improve while the same endpoint moves materially away on
    # the other segment coordinate.  Treat that as setup, not verified route
    # progress.  An oriented topology crossing is authoritative only for a
    # transport primitive; incidental pivot/alignment drift remains setup.
    consistent_segment_progress = bool(
        (local_goal_delta_m >= 0.05 and along_delta_m >= -0.02)
        or (along_delta_m >= 0.05 and local_goal_delta_m >= -0.02)
    )
    topology_transport_progress = bool(
        topology_advanced and action in _LINEAR_ACTIONS
    )
    effective_completed_prefix_after = (
        completed_prefix_after
        if topology_transport_progress
        else completed_prefix_before
    )
    transport_progress = bool(
        action in _LINEAR_ACTIONS
        and (consistent_segment_progress or topology_transport_progress)
    )
    return {
        "progress": bool(topology_transport_progress or transport_progress),
        "topology_advanced": topology_transport_progress,
        "raw_topology_crossing": bool(topology_advanced),
        "completed_prefix_before": completed_prefix_before,
        "completed_prefix_after": effective_completed_prefix_after,
        "local_goal_progress_m": round(local_goal_delta_m, 6),
        "along_segment_progress_m": round(along_delta_m, 6),
        "consistent_segment_progress": consistent_segment_progress,
        "transport_progress": transport_progress,
        "lifecycle_progress": False,
        "lifecycle_is_setup": action in _LIFECYCLE_ACTIONS,
    }


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _phase_anchor_regression(
    *,
    task: BoxCarryTask,
    after_state: dict[str, Any],
    phase_guard: dict[str, Any],
) -> dict[str, Any]:
    payload = after_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    try:
        x = float(payload["x"])
        y = float(payload["y"])
    except (KeyError, TypeError, ValueError):
        return {}
    anchors = phase_guard.get("phase_anchors", [])
    anchors = anchors if isinstance(anchors, list) else []
    try:
        active_prefix = int(phase_guard.get("maximum_completed_prefix", 0))
    except (TypeError, ValueError):
        active_prefix = 0
    anchor_prefixes: list[int] = []
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        try:
            prefix = int(anchor.get("completed_prefix", 0))
        except (TypeError, ValueError):
            continue
        if 0 < prefix <= active_prefix:
            anchor_prefixes.append(prefix)
    latest_prefix = max(anchor_prefixes, default=0)
    if latest_prefix == 0:
        return {}
    margin_m = 0.35
    route_gate_count = len(task.route_progress_gates)
    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue
        prefix = int(anchor.get("completed_prefix", 0))
        if prefix != latest_prefix:
            continue
        anchor_x = float(anchor.get("payload_x_m", x))
        anchor_y = float(anchor.get("payload_y_m", y))
        gate_progress: float | None = None
        candidate_progress: float | None = None
        if 1 <= prefix <= route_gate_count:
            gate = task.route_progress_gates[prefix - 1]
            normal_x, normal_y = _unit_gate_normal(gate)
            candidate_progress = normal_x * x + normal_y * y
            gate_progress = float(gate.threshold_m)
            # Retain the authored topology boundary, not the accepted action's
            # overshoot. A room-scale action can finish well beyond a gate;
            # requiring every later segment to preserve that accidental maximum
            # blocks valid motion after a corner whenever the outgoing segment is
            # not contained in the incoming gate's forward half-space.
            violated = candidate_progress < gate_progress - margin_m
        else:
            violated = bool(
                prefix >= route_gate_count + 1
                and hypot(x - task.target_center_x_m, y - task.target_center_y_m)
                > max(
                    task.target_xy_tolerance_m,
                    task.stage3_right_goal_tolerance_m,
                )
                + margin_m
            )
        if violated:
            evidence = {
                "violated": True,
                "completed_prefix": prefix,
                "anchor_checkpoint": anchor.get("checkpoint_after", ""),
                "anchor_payload_xy_m": [anchor_x, anchor_y],
                "candidate_payload_xy_m": [x, y],
                "retention_margin_m": margin_m,
            }
            if gate_progress is not None and candidate_progress is not None:
                evidence.update(
                    {
                        "completed_gate_threshold_m": gate_progress,
                        "candidate_gate_progress_m": candidate_progress,
                    }
                )
            return evidence
    return {}


def _active_phase_progress(
    *,
    task: BoxCarryTask,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    phase_guard: dict[str, Any],
) -> dict[str, Any]:
    before_payload = before_state.get("payload", {})
    after_payload = after_state.get("payload", {})
    if not isinstance(before_payload, dict) or not isinstance(after_payload, dict):
        return {"available": False}
    try:
        before_x = float(before_payload["x"])
        before_y = float(before_payload["y"])
        after_x = float(after_payload["x"])
        after_y = float(after_payload["y"])
    except (KeyError, TypeError, ValueError):
        return {"available": False}
    prefix = int(phase_guard.get("maximum_completed_prefix", 0))
    if prefix < len(task.route_progress_gates):
        gate = task.route_progress_gates[prefix]
        normal_x, normal_y = _unit_gate_normal(gate)
        before_value = normal_x * before_x + normal_y * before_y
        after_value = normal_x * after_x + normal_y * after_y
        threshold = float(gate.threshold_m)
        active_gate_crossed = bool(
            before_value + 1.0e-9 < threshold
            and after_value + 1.0e-9 >= threshold
        )
        name = f"oriented_gate:{gate.phase_name}"
    elif prefix == len(task.route_progress_gates):
        before_distance = hypot(
            before_x - task.target_center_x_m,
            before_y - task.target_center_y_m,
        )
        after_distance = hypot(
            after_x - task.target_center_x_m,
            after_y - task.target_center_y_m,
        )
        before_value = -before_distance
        after_value = -after_distance
        threshold = -max(
            task.target_xy_tolerance_m,
            task.stage3_right_goal_tolerance_m,
        )
        active_gate_crossed = bool(
            before_value + 1.0e-9 < threshold
            and after_value + 1.0e-9 >= threshold
        )
        name = "negative_target_distance"
    else:
        return {"available": False, "phase_prefix": prefix}
    return {
        "available": True,
        "phase_prefix": prefix,
        "progress_coordinate": name,
        "before": round(before_value, 6),
        "after": round(after_value, 6),
        "delta_m": round(after_value - before_value, 6),
        "threshold": round(threshold, 6),
        "active_gate_crossed": active_gate_crossed,
    }


def _manifest_route_states(
    task: BoxCarryTask,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    initial = manifest.get("initial_state", {})
    initial = initial if isinstance(initial, dict) else {}
    payload = initial.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    states: list[dict[str, Any]] = []
    try:
        states.append(
            {
                "x": float(payload["x"]),
                "y": float(payload["y"]),
                "yaw_deg": float(payload.get("yaw_deg", 0.0)),
                "lifecycle": "released" if payload.get("released", False) else "carrying",
                "phase_prefix": 0,
                **_formation_cycle_state({}),
            }
        )
    except (KeyError, TypeError, ValueError):
        pass
    yaw = float(states[-1]["yaw_deg"]) if states else 0.0
    trusted_prefix = int(states[-1].get("phase_prefix", 0)) if states else 0
    previous_route_progress_xy = (
        (float(states[-1]["x"]), float(states[-1]["y"])) if states else None
    )
    for step in manifest.get("steps", ()):
        if not isinstance(step, dict):
            continue
        if not _manifest_step_is_route_trajectory(step):
            continue
        measured = step.get("measured_outcome", {})
        measured = measured if isinstance(measured, dict) else {}
        try:
            yaw += float(measured.get("payload_step_signed_yaw_change_deg", 0.0))
            current_xy = (
                float(measured["payload_x_m"]),
                float(measured["payload_y_m"]),
            )
            route_progress_step = str(
                step.get("transition_outcome", "")
            ).strip() in {
                "ACCEPT_PROGRESS",
                "TASK_COMPLETE",
            }
            if route_progress_step:
                trusted_prefix = advance_trusted_route_prefix(
                    task,
                    prefix=trusted_prefix,
                    previous_route_progress_xy=previous_route_progress_xy,
                    current_route_progress_xy=current_xy,
                )
            states.append(
                {
                    "x": current_xy[0],
                    "y": current_xy[1],
                    "yaw_deg": yaw,
                    "lifecycle": "carrying",
                    "phase_prefix": trusted_prefix,
                    **_formation_cycle_state(measured),
                }
            )
            if route_progress_step:
                previous_route_progress_xy = current_xy
        except (KeyError, TypeError, ValueError):
            continue
    return states


def _manifest_step_is_route_trajectory(step: dict[str, Any]) -> bool:
    """Include every trusted physical checkpoint in oscillation evidence.

    Setup checkpoints remain ineligible for phase completion and positive
    memory, but excluding their poses lets forward/backward setup motion evade
    recurrence detection indefinitely.
    """

    outcome = str(step.get("transition_outcome", "")).strip()
    if outcome:
        return outcome in {"ACCEPT_SETUP", "ACCEPT_PROGRESS", "TASK_COMPLETE"}
    if "positive_route_memory_eligible" in step:
        return step.get("positive_route_memory_eligible") is True
    # Engineering manifests written before the six-state transition contract
    # have neither field. Keep them readable for diagnostics and migration.
    return True


def _candidate_route_state(
    report: EvaluationReport,
    state: dict[str, Any],
    prior: list[dict[str, Any]],
    *,
    phase_prefix: int | None,
) -> dict[str, Any] | None:
    payload = state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    try:
        x = float(report.metrics.get("payload_x_m", payload["x"]))
        y = float(report.metrics.get("payload_y_m", payload["y"]))
        yaw = float(payload.get("yaw_deg", prior[-1]["yaw_deg"] if prior else 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    prefix = (
        max(0, int(phase_prefix))
        if phase_prefix is not None
        else int(prior[-1].get("phase_prefix", 0))
        if prior
        else 0
    )
    return {
        "x": x,
        "y": y,
        "yaw_deg": yaw,
        "lifecycle": "released" if payload.get("released", False) else "carrying",
        "phase_prefix": prefix,
        **_formation_cycle_state(report.metrics),
    }


def _coarse_state_signature(
    task: BoxCarryTask,
    state: dict[str, Any],
) -> tuple[Any, ...]:
    x = float(state["x"])
    y = float(state["y"])
    phase = int(
        state.get("phase_prefix", _coordinate_phase_prefix(task, x=x, y=y))
    )
    return (
        phase,
        round(x / 0.25),
        round(y / 0.25),
        round(float(state["yaw_deg"]) / 15.0),
        str(state["lifecycle"]),
        state.get("formation_stagger_bin", "unknown"),
        state.get("formation_skew_bin", "unknown"),
        state.get("formation_heading_split_bin", "unknown"),
        state.get("formation_support_error_bin", "unknown"),
    )


def _formation_cycle_state(metrics: object) -> dict[str, int | str]:
    """Return coarse deterministic formation state for recurrence identity."""

    values = metrics if isinstance(metrics, dict) else {}

    def coarse(key: str, width: float) -> int | str:
        try:
            value = float(values[key])
        except (KeyError, TypeError, ValueError):
            return "unknown"
        if value != value:
            return "unknown"
        return round(value / width)

    return {
        "formation_stagger_bin": coarse(
            "robot_pair_longitudinal_stagger_m", 0.05
        ),
        "formation_skew_bin": coarse("robot_pair_formation_skew_deg", 5.0),
        "formation_heading_split_bin": coarse(
            "robot_heading_disagreement_after_deg", 5.0
        ),
        "formation_support_error_bin": coarse(
            "payload_support_center_error_m", 0.05
        ),
    }


def _coordinate_phase_prefix(task: BoxCarryTask, *, x: float, y: float) -> int:
    if hypot(x - task.target_center_x_m, y - task.target_center_y_m) <= max(
        task.target_xy_tolerance_m,
        task.stage3_right_goal_tolerance_m,
    ):
        return len(task.route_progress_gates) + 1
    prefix = 0
    while prefix < len(task.route_progress_gates):
        gate = task.route_progress_gates[prefix]
        normal_x, normal_y = _unit_gate_normal(gate)
        if normal_x * x + normal_y * y + 1.0e-9 < float(gate.threshold_m):
            break
        prefix += 1
    return prefix


def _unit_gate_normal(gate: Any) -> tuple[float, float]:
    normal_x = float(gate.normal_x)
    normal_y = float(gate.normal_y)
    magnitude = hypot(normal_x, normal_y)
    if magnitude <= 1.0e-9:
        raise ValueError(f"route progress gate {gate.phase_name!r} has zero normal")
    return normal_x / magnitude, normal_y / magnitude


def _last_action(trace: ExecutionTrace) -> str:
    for call in reversed(trace.skill_calls):
        if call.skill_name != "stop_all":
            return str(call.skill_name)
    return str(trace.skill_calls[-1].skill_name) if trace.skill_calls else ""


def _locked_turn_episode_reversal(
    *,
    before_state: dict[str, Any],
    trace: ExecutionTrace,
    action: str,
) -> dict[str, Any]:
    """Reject cancellation of an unfinished accepted signed turn episode."""

    episode = before_state.get("turn_episode", {})
    episode = episode if isinstance(episode, dict) else {}
    if (
        action not in _LOCAL_CORRECTION_ACTIONS
        or not bool(episode.get("active", False))
        or str(episode.get("purpose", "")).strip() != "route_turn"
        or bool(episode.get("switch_allowed", False))
    ):
        return {}
    expected = str(episode.get("direction", "")).strip().lower()
    actual = ""
    if action in {"turn_clockwise", "turn_counterclockwise"}:
        actual = action.removeprefix("turn_")
    else:
        for call in reversed(trace.skill_calls):
            if call.skill_name == action:
                actual = str(call.parameters.get("turn_direction", "")).strip().lower()
                break
    if {expected, actual} != {"clockwise", "counterclockwise"}:
        return {}
    return {
        "expected_turn_direction": expected,
        "candidate_turn_direction": actual,
        "switch_allowed": False,
        "episode_checkpoint_id": str(
            episode.get("latest_checkpoint_id", "")
        ).strip(),
    }


__all__ = [
    "TransitionDecision",
    "TransitionOutcome",
    "apply_transition_decision",
    "detect_state_progress_cycle",
    "transition_is_route_progress",
    "transition_outcome_from_report",
    "verify_transition",
]
