"""Physics-authoritative evaluator for box-carry Code-as-Policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .contracts import BoxCarryTask, EvaluationReport, ExecutionTrace
from .failure_taxonomy import annotate_evaluation_report
from .simulation_diagnostics import (
    MAX_STRAIGHT_HEADING_ERROR_DEG,
    BoxCarryEvaluator as SimulationDiagnosticEvaluator,
    _active_goal_xy,
    _monitored_physical_failure,
    _route_regression_evidence,
)
from .visual_verifier import VisualVerifierReport
from praxis_coop.sonic.safety import CarrySafetyProfile, get_carry_safety_profile


_TRANSITION_SEMANTIC_FAILURES = frozenset(
    {
        "no_payload_progress",
        "goal_regression",
        "route_regression",
        "straightness_regression",
        "alignment_regression",
    }
)


@dataclass
class BoxCarryEvaluator:
    """Fuse deterministic physics gates with advisory visual route evidence.

    MuJoCo runtime state is authoritative for payload support, floor/wall
    contact, robot stability, spacing, and rollback. The visual model supplies
    global-scene semantics and can establish a route regression only when the
    physical evaluator corroborates that the commanded rollout actually moved.

    ``require_visual`` prevents a physically safe endpoint from being committed
    without a fresh global frame. It never masks an already observed physical
    failure.
    """

    runtime_state_file: Path
    safety_profile: str | CarrySafetyProfile = "task"
    require_visual: bool = False

    def evaluate(
        self,
        *,
        task: BoxCarryTask,
        trace: ExecutionTrace,
        visual_summary: str = "",
        visual_report: VisualVerifierReport | None = None,
        before_state: dict[str, Any] | None = None,
        state_override: dict[str, Any] | None = None,
        stepwise: bool = False,
        straight_heading_reference_rad: float | None = None,
        turn_episode: dict[str, Any] | None = None,
        route_phase_id: str = "",
        allow_pending_visual: bool = False,
    ) -> EvaluationReport:
        diagnostic = SimulationDiagnosticEvaluator(
            runtime_state_file=self.runtime_state_file,
            safety_profile=self.safety_profile,
        ).evaluate(
            task=task,
            trace=trace,
            visual_summary=visual_summary,
            visual_report=visual_report,
            before_state=before_state,
            state_override=state_override,
            stepwise=stepwise,
            straight_heading_reference_rad=straight_heading_reference_rad,
            turn_episode=turn_episode,
            route_phase_id=route_phase_id,
        )
        diagnostic = _defer_task_transition_semantics(
            diagnostic,
            safety_profile=self.safety_profile,
            stepwise=stepwise,
        )
        diagnostic = _with_decision_policy(diagnostic)

        if not self.require_visual or visual_report is not None:
            return diagnostic

        # Preliminary evaluation intentionally runs before the post-action
        # image is captured. Preserve any physical rejection immediately; a
        # pending camera frame is allowed only for a currently safe endpoint.
        if not diagnostic.success:
            return diagnostic
        if allow_pending_visual:
            return annotate_evaluation_report(
                replace(
                    diagnostic,
                    evidence={
                        **diagnostic.evidence,
                        "visual_verifier": {"status": "pending"},
                    },
                )
            )
        return annotate_evaluation_report(
            replace(
                diagnostic,
                success=False,
                failure_type="manual_review_required",
                failed_subtask=_last_skill_name(trace),
                evidence={
                    **diagnostic.evidence,
                    "visual_verifier": {
                        "status": "missing",
                        "reason": "fresh global frame required before checkpoint commit",
                    },
                },
                recommendations=(
                    "capture the post-action global frame before accepting this endpoint",
                ),
            )
        )


def _with_decision_policy(report: EvaluationReport) -> EvaluationReport:
    evidence = {
        **report.evidence,
        "decision_policy": {
            "mode": "physics_authoritative_multimodal",
            "physics_decides": [
                "payload_support_and_floor_contact",
                "robot_stability",
                "wall_contact",
                "team_spacing",
                "rollback",
            ],
            "visual_supplies": [
                "global_route_semantics",
                "visible_passage_connectivity",
                "target_and_posture_context",
            ],
            "visual_cannot_override_fresh_physics": True,
        },
    }
    return annotate_evaluation_report(replace(report, evidence=evidence))


def _defer_task_transition_semantics(
    report: EvaluationReport,
    *,
    safety_profile: str | CarrySafetyProfile,
    stepwise: bool,
) -> EvaluationReport:
    """Let one transition state machine own task-progress acceptance.

    ``simulation_diagnostics`` still computes all route, heading, and alignment
    measurements. In the formal task profile those measurements must not cause
    an early rollback before ``transition_verifier`` can evaluate corridor and
    topology semantics. Strict/debug mode intentionally retains the legacy
    single-layer rejections.
    """

    profile = get_carry_safety_profile(safety_profile)
    if (
        not stepwise
        or profile.name != "task"
        or report.success
        or report.failure_type not in _TRANSITION_SEMANTIC_FAILURES
    ):
        return report
    deferred = {
        "failure_type": report.failure_type,
        "failure_level": report.failure_level,
        "failed_subtask": report.failed_subtask,
        "diagnosis": report.diagnosis,
        "recommendations": list(report.recommendations),
        "decision_owner": "transition_verifier",
    }
    return annotate_evaluation_report(
        replace(
            report,
            success=True,
            failure_type="none",
            failed_subtask="",
            evidence={
                **report.evidence,
                "deferred_transition_semantic_diagnostic": deferred,
            },
            recommendations=(
                "pass physical measurements to the transition verifier",
            ),
        )
    )


def _last_skill_name(trace: ExecutionTrace) -> str:
    for item in reversed(trace.skill_calls):
        if item.skill_name != "stop_all":
            return item.skill_name
    return trace.skill_calls[-1].skill_name if trace.skill_calls else ""


__all__ = [
    "BoxCarryEvaluator",
    "MAX_STRAIGHT_HEADING_ERROR_DEG",
    "_active_goal_xy",
    "_monitored_physical_failure",
    "_route_regression_evidence",
]
