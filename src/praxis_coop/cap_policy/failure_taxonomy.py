"""Verifier-grounded failure levels for cooperative script evolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .contracts import EvaluationReport, FailureLevel


SYSTEM_FAILURES = frozenset(
    {"script_error", "runtime_state_missing", "manual_review_required"}
)
STRATEGY_FAILURES = frozenset(
    {
        "goal_regression",
        "route_regression",
        "straightness_regression",
        "alignment_regression",
        "payload_wall_contact",
        "payload_not_on_target",
        "robots_not_clear",
    }
)
GENERATED_SCRIPT_FAILURES = frozenset(
    {
        "formation_desync",
        "payload_drop",
        "payload_not_supported",
        "payload_slip",
        "payload_unstable",
    }
)
EXECUTION_FAILURES = frozenset(
    {"payload_not_active", "robot_fall", "timeout"}
)


@dataclass(frozen=True)
class FailureClassification:
    """One deterministic repair scope derived from verifier evidence."""

    level: FailureLevel
    diagnosis: str
    allowed_repairs: tuple[str, ...]
    forbidden_repairs: tuple[str, ...]
    promotable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "diagnosis": self.diagnosis,
            "allowed_repairs": list(self.allowed_repairs),
            "forbidden_repairs": list(self.forbidden_repairs),
            "promotable": self.promotable,
        }


def classify_failure(
    failure_type: str,
    *,
    metrics: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    failed_subtask: str = "",
) -> FailureClassification:
    """Classify what the next evolution step is allowed to change."""

    failure = str(failure_type or "none")
    metrics = metrics if isinstance(metrics, dict) else {}
    evidence = evidence if isinstance(evidence, dict) else {}
    subtask = str(failed_subtask or "").strip()
    if failure == "none":
        return FailureClassification(
            level="none",
            diagnosis="The external-camera evaluator accepted this bounded rollout.",
            allowed_repairs=("select_next_bounded_subtask",),
            forbidden_repairs=("generalize_beyond_comparable_evidence",),
        )
    if failure in SYSTEM_FAILURES:
        return FailureClassification(
            level="system",
            diagnosis=(
                f"{failure} is an infrastructure, model-code, or evidence-availability failure; "
                "it does not evaluate a physical cooperation strategy."
            ),
            allowed_repairs=("repair_runtime_or_generated_code", "repeat_preflight_gate"),
            forbidden_repairs=("promote_to_dynamic_memory", "physical_strategy_search"),
            promotable=False,
        )
    if failure == "turn_not_ready":
        return FailureClassification(
            level="script",
            diagnosis=(
                "The exact model-authored pivot failed a live complete-team swept-"
                "clearance check before dispatch. This is checkpoint-local candidate "
                "rejection, not a physical strategy, coordination, or execution "
                "failure, and not a Host-selected route."
            ),
            allowed_repairs=(
                "re_orchestrate_a_different_bounded_subtask",
                "let_model_roles_select_the_replacement",
            ),
            forbidden_repairs=(
                "repeat_the_same_infeasible_executable_candidate",
                "lock_hierarchical_mutation_to_the_unexecuted_parent",
                "count_as_physical_iteration",
                "host_select_replacement_direction_roles_or_primitive",
            ),
            promotable=False,
        )
    if failure == "payload_unstable" and subtask in {
        "turn_clockwise",
        "turn_counterclockwise",
    }:
        return FailureClassification(
            level="execution",
            diagnosis=(
                f"The {subtask} strategy and synchronized role relationship remain viable, "
                "but this bounded turn exceeded the stable payload execution envelope."
            ),
            allowed_repairs=(
                "keep_turn_direction_and_coordination_strategy",
                "reduce_exactly_one_bounded_turn_parameter_prefer_yaw_deg_then_yaw_rate",
            ),
            forbidden_repairs=(
                "increase_turn_magnitude_or_rate",
                "change_turn_direction_or_coordination_strategy",
                "change_multiple_execution_parameters_at_once",
                "change_unrelated_strategy_fields",
            ),
        )
    if failure in GENERATED_SCRIPT_FAILURES:
        return _generated_script_classification(failure, evidence=evidence)
    if failure in STRATEGY_FAILURES:
        return _strategy_classification(failure)
    if failure in EXECUTION_FAILURES:
        return _execution_classification(failure)
    if failure == "no_payload_progress":
        if subtask == "coordinated_alignment":
            return FailureClassification(
                level="script",
                diagnosis=(
                    "The alignment transaction produced no measurable correction; "
                    "its executable axis and moving/reference relationship are wrong "
                    "for this checkpoint."
                ),
                allowed_repairs=(
                    "rewrite_alignment_axis_or_moving_reference_relationship",
                    "select_a_different_coordination_primitive",
                ),
                forbidden_repairs=(
                    "change_alignment_delta_only",
                    "repeat_same_alignment_structure",
                ),
            )
        if bool(evidence.get("linear_no_progress_at_verified_ceiling")):
            return FailureClassification(
                level="strategy",
                diagnosis=(
                    "The largest verified linear transaction produced negligible "
                    "payload motion, so the current body-frame direction or maneuver "
                    "family is blocked or unsuitable."
                ),
                allowed_repairs=(
                    "change_motion_objective_or_direction",
                    "change_coordination_primitive_for_current_geometry",
                ),
                forbidden_repairs=(
                    "magnitude_only_retry",
                    "increase_speed_or_distance",
                    "repeat_same_strategy_signature",
                ),
            )
        displacement = _number(
            metrics.get(
                "payload_step_displacement_xy_m",
                metrics.get("payload_step_displacement_m", 0.0),
            )
        )
        required = _number(metrics.get("required_step_displacement_m", 0.0))
        if required > 0.0 and displacement >= required:
            return FailureClassification(
                level="strategy",
                diagnosis=(
                    "The payload moved measurably but did not reduce active-goal distance; "
                    "the current motion objective or body-frame direction is not useful here."
                ),
                allowed_repairs=(
                    "change_motion_objective_or_direction",
                    "change_coordination_strategy_if_geometry_requires_it",
                ),
                forbidden_repairs=(
                    "magnitude_only_retry",
                    "repeat_same_strategy_signature",
                ),
            )
        return FailureClassification(
            level="execution",
            diagnosis=(
                "The selected action produced less measurable payload motion than the verifier "
                "requires while the strategy has not yet been disproven."
            ),
            allowed_repairs=(
                "keep_validated_strategy",
                "tune_one_bounded_execution_parameter",
                "select_full_verified_magnitude_directly_when_clearance_is_stable",
            ),
            forbidden_repairs=(
                "change_strategy_before_execution_budget_is_exhausted",
                "change_multiple_execution_parameters_at_once",
            ),
        )
    return FailureClassification(
        level="system",
        diagnosis=f"{failure} has no trusted failure-level mapping.",
        allowed_repairs=("inspect_verifier_contract",),
        forbidden_repairs=("promote_to_dynamic_memory",),
        promotable=False,
    )


def classify_report(report: EvaluationReport) -> FailureClassification:
    return classify_failure(
        report.failure_type,
        metrics=report.metrics,
        evidence=report.evidence,
        failed_subtask=report.failed_subtask,
    )


def annotate_evaluation_report(report: EvaluationReport) -> EvaluationReport:
    classification = classify_report(report)
    return replace(
        report,
        failure_level=classification.level,
        diagnosis=classification.diagnosis,
        allowed_repairs=classification.allowed_repairs,
        forbidden_repairs=classification.forbidden_repairs,
    )


def _strategy_classification(failure: str) -> FailureClassification:
    diagnoses = {
        "goal_regression": (
            "The selected body-frame motion moved the payload away from the active goal."
        ),
        "route_regression": (
            "The executed motion continued into visibly blocked space or retreated from both "
            "the target and connected free space without visible detour evidence."
        ),
        "straightness_regression": (
            "A non-turn transport segment changed the loaded team's reference heading or "
            "left its intended longitudinal/lateral axis."
        ),
        "alignment_regression": (
            "The requested relative correction moved the team without improving any "
            "measured alignment objective."
        ),
        "payload_wall_contact": (
            "The external camera shows a robot or payload colliding with or stuck at a wall."
        ),
        "payload_not_on_target": (
            "The transport or placement strategy has not aligned the payload center with the target."
        ),
        "robots_not_clear": (
            "The task strategy omitted a valid post-placement release and exit phase."
        ),
    }
    if failure == "straightness_regression":
        return FailureClassification(
            level="strategy",
            diagnosis=diagnoses[failure],
            allowed_repairs=(
                "restore_last_straight_checkpoint",
                "apply_one_synchronized_bounded_heading_correction",
                "apply_one_relative_yaw_alignment_for_robot_heading_split",
            ),
            forbidden_repairs=(
                "continue_translation_while_heading_is_skewed",
                "change_translation_speed_or_distance",
                "reinterpret_lateral_translation_as_a_turn",
            ),
        )
    return FailureClassification(
        level="strategy",
        diagnosis=diagnoses[failure],
        allowed_repairs=(
            "change_motion_objective_or_direction",
            "change_coordination_strategy",
            "change_transport_formation_from_current_evidence",
        ),
        forbidden_repairs=(
            "magnitude_only_retry",
            "repeat_same_strategy_signature",
            "increase_speed_or_distance_first",
        ),
    )


def _generated_script_classification(
    failure: str,
    *,
    evidence: dict[str, Any],
) -> FailureClassification:
    diagnoses = {
        "formation_desync": (
            "The generated front/back executable calls produced incompatible physical "
            "displacements and broke the loaded formation."
        ),
        "payload_drop": (
            "The external camera shows that the generated transaction left the payload "
            "resting on the floor."
        ),
        "payload_not_supported": (
            "The generated executable transaction caused at least one robot to lose valid "
            "hand/forearm support."
        ),
        "payload_slip": (
            "The generated executable transaction moved the payload relative to the live "
            "dual-arm support center."
        ),
        "payload_unstable": (
            "The generated executable transaction produced unsafe payload tilt or angular motion."
        ),
    }
    diagnosis = diagnoses[failure]
    route_evidence = evidence.get("route_direction_verification", {})
    if (
        isinstance(route_evidence, dict)
        and route_evidence.get("decision") == "reject_executed_route_direction"
    ):
        diagnosis += (
            " Verifier route evidence also rejects this semantic direction without "
            "reclassifying the physical script failure as strategy-only."
        )
    return FailureClassification(
        level="script",
        diagnosis=diagnosis,
        allowed_repairs=(
            "stabilize_or_reestablish_support_before_motion",
            "identify_the_failing_executable_call_or_parameter_relation",
            "rewrite_the_executable_transaction_from_current_physics_evidence",
            "preserve_the_semantic_goal_unless_route_evidence_disproves_it",
        ),
        forbidden_repairs=(
            "text_only_metadata_or_rationale_change",
            "repeat_same_executable_ast",
            "blind_role_or_coordination_switch_without_a_script_diagnosis",
        ),
    )


def _execution_classification(failure: str) -> FailureClassification:
    diagnoses = {
        "payload_not_active": "The initialization event has not established an active payload.",
        "robot_fall": "The low-level execution left the valid upright operating state.",
        "timeout": "The selected bounded action did not complete within its execution window.",
    }
    return FailureClassification(
        level="execution",
        diagnosis=diagnoses[failure],
        allowed_repairs=(
            "keep_validated_strategy",
            "repair_initialization_or_execution_envelope",
            "tune_one_bounded_execution_parameter",
        ),
        forbidden_repairs=(
            "change_strategy_without_strategy_evidence",
            "increase_magnitude_after_instability",
        ),
    )


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "FailureClassification",
    "annotate_evaluation_report",
    "classify_failure",
    "classify_report",
]
