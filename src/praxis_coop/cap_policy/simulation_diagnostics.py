"""Deterministic physics verifier for box-carry rollouts.

The public fusion entry point lives in :mod:`.evaluator`, while this module
owns the MuJoCo/runtime-state acceptance gates and rollback evidence. Compact
metrics may be projected to the planning Agent; raw contacts stay local.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2, cos, degrees, hypot, sin
from pathlib import Path
from typing import Any

from .contracts import BoxCarryTask, EvaluationReport, ExecutionTrace
from .failure_taxonomy import annotate_evaluation_report
from .local_navigation import (
    LOCAL_NAVIGATION_CONTRACT_KEY,
    evaluate_local_navigation_transition,
)
from .coordination_specs import (
    COORDINATED_ALIGNMENT,
    COORDINATED_ARC_TRANSPORT,
    COORDINATED_LINEAR_TRANSPORT,
    COORDINATED_PIVOT,
    LONGITUDINAL_ALIGNMENT_TRIGGER_M,
    MAX_LONGITUDINAL_ALIGNMENT_DELTA_M,
    MIN_LONGITUDINAL_ALIGNMENT_DELTA_M,
    VERIFIED_LOADED_LINEAR_MAX_M,
)
from .visual_verifier import VisualVerifierReport
from praxis_coop.sonic.safety import (
    CarrySafetyProfile,
    PAYLOAD_TOP_FACE_UP_LIMIT_DEG,
    get_carry_safety_profile,
)
from praxis_coop.sonic.upper_body import PAYLOAD_RECOVERY_POSE_NAME


LINEAR_PROGRESS_ACTIONS = frozenset(
    {
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
        COORDINATED_LINEAR_TRANSPORT,
    }
)
CONTINUOUS_TURN_ACTIONS = frozenset(
    {
        COORDINATED_ARC_TRANSPORT,
        COORDINATED_PIVOT,
        "turn_clockwise",
        "turn_counterclockwise",
    }
)
STATIC_TURN_ACTIONS = frozenset({"turn_clockwise", "turn_counterclockwise"})
STATIONARY_LOADED_TURN_ACTIONS = STATIC_TURN_ACTIONS | {COORDINATED_PIVOT}
ROUTE_TRANSLATION_ACTIONS = LINEAR_PROGRESS_ACTIONS | CONTINUOUS_TURN_ACTIONS
MIN_MEANINGFUL_TRANSLATION_M = 0.005
MIN_MEANINGFUL_YAW_DEG = 2.0
MIN_ALIGNMENT_HEADING_IMPROVEMENT_DEG = 0.5
MIN_ALIGNMENT_FORMATION_SKEW_IMPROVEMENT_DEG = 0.5
MIN_ALIGNMENT_AXIS_IMPROVEMENT = 0.005
MIN_ALIGNMENT_SUPPORT_CENTER_IMPROVEMENT_M = 0.005
MIN_LONGITUDINAL_STAGGER_IMPROVEMENT_M = 0.005
MIN_TURN_ALIGNMENT_IMPROVEMENT = 0.02
PAYLOAD_INWARD_RECOVERY_ACTION = "carry_recover_payload_inward"
FORMATION_SYNC_REPAIR_PROBE_KIND = "formation_sync_repair_probe"
FORMATION_SYNC_REPAIR_MAX_NORMALIZED_MISMATCH = 0.20
FORMATION_SYNC_REPAIR_MIN_PROGRESS_M = 0.05
POST_RELEASE_MIN_CROUCH_FRACTION = 0.80
POST_RELEASE_UPRIGHT_MIN_PELVIS_HEIGHT_M = 0.68

# One compact, non-CLI straight-transport contract. Lateral translation is a
# valid axis; it is not a turn and must preserve the same body heading.
MAX_STRAIGHT_HEADING_ERROR_DEG = 3.0
MAX_STRAIGHT_PATH_ERROR_DEG = 10.0
MIN_STRAIGHT_PATH_AUDIT_M = 0.20
# Keep one non-CLI hard boundary for an obviously wrong linear trajectory.
# Smaller heading, path, and formation errors remain diagnostics in the task
# profile while both robots still legally support the payload. The strict
# ablation retains the conservative diagnostic thresholds as hard failures.
MAX_SEVERE_LINEAR_DEVIATION_DEG = 30.0
@dataclass
class BoxCarryEvaluator:
    """Evaluate authoritative MuJoCo state and corroborated visual route risk."""

    runtime_state_file: Path
    safety_profile: str | CarrySafetyProfile = "task"

    @property
    def safety(self) -> CarrySafetyProfile:
        return get_carry_safety_profile(self.safety_profile)

    @property
    def max_robot_step_vector_desync_m(self) -> float:
        return self.safety.warning_robot_step_desync_m

    @property
    def max_payload_step_yaw_drift_deg(self) -> float:
        return self.safety.warning_payload_yaw_drift_deg

    @property
    def max_payload_formation_slip_m(self) -> float:
        return self.safety.warning_payload_formation_slip_m

    @property
    def max_payload_support_center_error_m(self) -> float:
        return self.safety.max_support_center_error_m

    @property
    def min_robot_payload_translation_ratio(self) -> float:
        return self.safety.warning_min_translation_tracking_ratio

    @property
    def min_payload_support_force_per_robot_n(self) -> float:
        return self.safety.min_support_force_per_robot_n

    @property
    def min_payload_support_force_balance_ratio(self) -> float:
        return self.safety.min_support_force_balance_ratio

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
    ) -> EvaluationReport:
        visual_summary = visual_summary or _visual_summary(visual_report)
        monitored_failure = _monitored_physical_failure(trace)
        if trace.status == "script_error" and monitored_failure is None:
            return annotate_evaluation_report(EvaluationReport(
                success=False,
                failure_type="script_error",
                failed_subtask=_last_skill_name(trace),
                evidence={"error": trace.error, **_visual_evidence(visual_report)},
                recommendations=("repair generated Python script before running robot again",),
                visual_summary=visual_summary,
            ))

        state = state_override if state_override is not None else self._read_state()
        if not state:
            return annotate_evaluation_report(EvaluationReport(
                success=False,
                failure_type="runtime_state_missing",
                evidence={"runtime_state_file": str(self.runtime_state_file), **_visual_evidence(visual_report)},
                recommendations=("start Term1 with --runtime-state-file matching the policy output directory",),
                visual_summary=visual_summary,
            ))

        payload = state.get("payload", {}) if isinstance(state.get("payload"), dict) else {}
        before_payload = (
            before_state.get("payload", {})
            if isinstance(before_state, dict)
            and isinstance(before_state.get("payload"), dict)
            else {}
        )
        robots = state.get("robots", {}) if isinstance(state.get("robots"), dict) else {}
        contacts = payload.get("contacts", {}) if isinstance(payload.get("contacts"), dict) else {}
        all_robot_contacts = tuple(contacts.get("robots", ()))
        robot_contacts = tuple(contacts.get("support_robots", ()))
        support_geometries = tuple(contacts.get("support_geometries", ()))
        wall_contacts = tuple(contacts.get("walls", ()))

        synchronization_metrics, synchronization_evidence = _synchronization_metrics(
            task,
            before_state,
            state,
        )
        straightness_metrics, straightness_evidence = _translation_straightness_metrics(
            trace,
            before_state,
            state,
            reference_yaw_rad=straight_heading_reference_rad,
        )
        support_center_metrics, support_center_evidence = _payload_support_center_metrics(payload)
        (
            before_support_center_metrics,
            before_support_center_evidence,
        ) = _payload_support_center_metrics(before_payload)
        support_center_comparison_available = bool(
            support_center_evidence["available"]
            and before_support_center_evidence["available"]
        )
        support_center_error_before = before_support_center_metrics[
            "payload_support_center_error_m"
        ]
        support_center_error_after = support_center_metrics[
            "payload_support_center_error_m"
        ]
        support_center_improvement = (
            support_center_error_before - support_center_error_after
            if support_center_comparison_available
            else 0.0
        )
        support_center_metrics.update(
            {
                "payload_support_center_error_before_m": support_center_error_before,
                "payload_support_center_error_improvement_m": support_center_improvement,
            }
        )
        support_center_evidence.update(
            {
                "before_available": bool(before_support_center_evidence["available"]),
                "before_error_m": support_center_error_before,
                "improvement_m": support_center_improvement,
            }
        )
        support_force_metrics, support_force_evidence = _payload_support_force_metrics(contacts)
        robot_pair_xy_separation = _robot_pair_xy_separation(robots)
        metrics = {
            "payload_z_m": _float(payload.get("z"), 0.0),
            "payload_x_m": _float(payload.get("x"), 0.0),
            "payload_y_m": _float(payload.get("y"), 0.0),
            "wall_contact_count": float(len(wall_contacts)),
            "payload_robot_contact_count": float(len(set(robot_contacts))),
            "payload_support_surface_contact_count": float(len(set(support_geometries))),
            "min_robot_pelvis_height_m": _min_robot_height(robots),
            "robot_pair_xy_separation_m": (
                0.0 if robot_pair_xy_separation is None else robot_pair_xy_separation
            ),
            "forward_progress_m": _estimate_forward_progress(state),
            "payload_tilt_deg": _float(payload.get("tilt_deg"), 0.0),
            "payload_top_surface_z_range_m": _float(
                payload.get("top_surface_corner_z_range_m"), 0.0
            ),
            "payload_linear_speed_mps": _float(payload.get("linear_speed_mps"), 0.0),
            "payload_angular_speed_radps": _float(payload.get("angular_speed_radps"), 0.0),
            "payload_step_displacement_m": _payload_step_displacement(before_state, state),
            "payload_target_progress_m": _payload_target_progress(task, before_state, state),
            "payload_final_target_progress_m": _payload_final_target_progress(
                task,
                before_state,
                state,
            ),
            "payload_body_forward_support_offset_m": _float(
                payload.get("support_center_signed_longitudinal_offset_m"),
                0.0,
            ),
            "payload_body_forward_support_offset_before_m": _float(
                before_payload.get("support_center_signed_longitudinal_offset_m"),
                0.0,
            ),
            **synchronization_metrics,
            **straightness_metrics,
            **support_center_metrics,
            **support_force_metrics,
        }
        target_metrics = _target_metrics(task, payload, robots)
        metrics.update(target_metrics)
        before_robots = (
            before_state.get("robots", {})
            if isinstance(before_state, dict) and isinstance(before_state.get("robots"), dict)
            else {}
        )
        before_target_metrics = _target_metrics(task, before_payload, before_robots)
        lifecycle = _post_release_lifecycle_state(state)
        before_lifecycle = _post_release_lifecycle_state(before_state)
        robots_crouched = _robots_in_post_release_crouch(state)
        before_robots_crouched = _robots_in_post_release_crouch(before_state)
        robots_upright = _robots_upright_after_release(state)
        metrics.update(
            {
                "post_release_crouch_latched": float(lifecycle["crouch_latched"]),
                "post_release_crouched_retreat_completed": float(
                    lifecycle["crouched_retreat_completed"]
                ),
                "post_release_stand_recovered": float(
                    lifecycle["stand_recovered"]
                ),
                "both_robots_post_release_crouched": float(robots_crouched),
                "both_robots_post_release_upright": float(robots_upright),
            }
        )
        metrics["robot_exit_progress_m"] = float(
            metrics["min_robot_target_distance_m"]
            - _min_robot_target_distance(task, before_robots)
        ) if before_robots else 0.0
        active_goal_metrics = _active_goal_metrics(task, payload)
        metrics.update(active_goal_metrics)
        verification_contract = (
            trace.verification_contract
            if isinstance(trace.verification_contract, dict)
            else {}
        )
        local_navigation_contract = verification_contract.get(
            LOCAL_NAVIGATION_CONTRACT_KEY,
            {},
        )
        local_navigation_contract = (
            local_navigation_contract
            if isinstance(local_navigation_contract, dict)
            else {}
        )
        evidence: dict[str, Any] = {
            "runtime_state_file": str(self.runtime_state_file),
            "route_discovery": bool(task.route_discovery),
            "payload": payload,
            "robot_contacts": robot_contacts,
            "all_robot_contacts": all_robot_contacts,
            "support_geometries": support_geometries,
            "wall_contacts": wall_contacts,
            "target": _target_evidence(task),
            "active_stage_goal": _active_goal_evidence(task, payload),
            "formation_synchronization": synchronization_evidence,
            "straight_transport": straightness_evidence,
            "payload_support_center": support_center_evidence,
            "payload_support_force": support_force_evidence,
            "upper_body_pose_lock": state.get("upper_body_pose_lock", {}),
            "robot_pair_spacing": {
                "available": robot_pair_xy_separation is not None,
                "measured_xy_separation_m": robot_pair_xy_separation,
                "minimum_xy_separation_m": task.min_robot_xy_separation_m,
            },
            "post_release_lifecycle": {
                "before": before_lifecycle,
                "after": lifecycle,
                "before_both_robots_crouched": before_robots_crouched,
                "after_both_robots_crouched": robots_crouched,
                "after_both_robots_upright": robots_upright,
                "before_robots_clear": bool(
                    before_target_metrics["robots_clear_after_place"]
                ),
                "before_robots_clear_for_stand": bool(
                    before_target_metrics["robots_clear_for_stand"]
                ),
                "after_robots_clear": bool(metrics["robots_clear_after_place"]),
            },
            "script_status": trace.status,
            "skill_call_statuses": [
                {"skill": item.skill_name, "status": item.status, "artifact": item.artifact_path}
                for item in trace.skill_calls
            ],
            **_visual_evidence(visual_report),
        }
        if local_navigation_contract:
            evidence["local_navigation_contract"] = local_navigation_contract
        evidence["missing_optional_payload_state_fields"] = sorted(
            {
                "tilt_deg",
                "top_surface_corner_z_range_m",
                "linear_speed_mps",
                "angular_speed_radps",
            }
            - set(payload)
        )
        payload_on_target = bool(metrics["payload_on_target"])
        payload_released_on_target = bool(metrics["payload_released_on_target"])
        post_release_sequence_complete = bool(
            lifecycle["crouched_retreat_completed"]
            and lifecycle["stand_recovered"]
            and robots_upright
            and metrics["robots_clear_after_place"]
        )
        metrics["post_release_sequence_complete"] = float(
            post_release_sequence_complete
        )
        if task.route_discovery:
            task_complete = payload_on_target and (
                not task.require_robots_exit or post_release_sequence_complete
            )
        elif task.task_stage in {"stage1_straight", "stage2_corridor", "stage3_right"}:
            task_complete = bool(metrics["active_stage_goal_reached"])
        else:
            task_complete = payload_on_target and (
                not task.require_robots_exit or post_release_sequence_complete
            )
        metrics["task_complete"] = float(task_complete)

        if task.require_payload_active and not bool(payload.get("active", False)):
            return _fail(
                "payload_not_active",
                "place_payload_on_arms",
                metrics,
                evidence,
                "place payload only after Term1 reports stable dropped robots and carry arm lock",
                visual_summary,
            )
        if monitored_failure is not None and (
            trace.status == "script_error"
            or any(item.status.startswith("failed") for item in trace.skill_calls)
        ):
            # Preserve the first online safety trip as the causal failure. The
            # free payload may continue falling after the publisher aborts, but
            # that later endpoint must not replace a spacing/formation trigger
            # and incorrectly blacklist the motion objective.
            evidence["monitored_physical_failure"] = monitored_failure
            return _fail(
                monitored_failure,
                _last_action_skill(trace),
                metrics,
                evidence,
                "SONIC transport safety monitor aborted this physical step; restore and evolve its motion parameters",
                visual_summary,
            )
        if metrics["min_robot_pelvis_height_m"] < task.min_robot_pelvis_height_m:
            return _fail(
                "robot_fall",
                _last_skill_name(trace),
                metrics,
                evidence,
                "stop experiment, recover standing stability before continuing transport",
                visual_summary,
            )
        if (
            robot_pair_xy_separation is not None
            and robot_pair_xy_separation < task.min_robot_xy_separation_m
        ):
            return _fail(
                "formation_desync",
                _last_action_skill(trace),
                metrics,
                evidence,
                (
                    "restore the previous checkpoint and change the collaboration relation; "
                    "the two pelvis roots must remain safely separated during loaded motion"
                ),
                visual_summary,
            )
        if (
            metrics["payload_z_m"] < task.min_payload_z_m
            and not payload_released_on_target
        ):
            return _fail(
                "payload_drop",
                _last_skill_name(trace),
                metrics,
                evidence,
                "raise payload_z or reduce transport speed before retrying",
                visual_summary,
            )
        if (
            self.safety.name == "strict"
            and
            metrics["payload_tilt_deg"] >= PAYLOAD_TOP_FACE_UP_LIMIT_DEG
            and not payload_released_on_target
        ):
            return _fail(
                "payload_unstable",
                _last_action_skill(trace),
                metrics,
                evidence,
                (
                    "restore the previous checkpoint; the payload top face "
                    "overturned while still touching the robots"
                ),
                visual_summary,
            )
        if (
            self.safety.name == "strict"
            and (
            metrics["payload_tilt_deg"] > self.safety.hard_max_tilt_deg
            or metrics["payload_top_surface_z_range_m"]
            > self.safety.max_top_surface_z_range_m
            )
        ):
            return _fail(
                "payload_unstable",
                _last_action_skill(trace),
                metrics,
                evidence,
                "restore the previous checkpoint and reduce the current motion segment or resynchronize support",
                visual_summary,
            )
        if (
            self.safety.name == "task"
            and metrics["payload_tilt_deg"] > self.safety.settled_max_tilt_deg
        ):
            _add_quality_warning(
                evidence,
                "payload_tilt",
                (
                    "payload remains legally supported but exceeds the preferred "
                    "settled tilt; preserve the checkpoint only as a recoverable "
                    "physical state, not proof of route or phase completion"
                ),
            )
        if (
            self.safety.name == "task"
            and metrics["payload_top_surface_z_range_m"]
            > self.safety.max_top_surface_z_range_m
        ):
            _add_quality_warning(
                evidence,
                "payload_surface_height_span",
                (
                    "payload remains off the floor with bilateral support, but its "
                    "surface-height span indicates a degraded carry posture"
                ),
            )
        if task.require_each_robot_payload_contact:
            missing = sorted(set(task.robot_ids) - set(robot_contacts))
            if missing and not payload_released_on_target:
                evidence["missing_payload_contact_robots"] = missing
                return _fail(
                    "payload_not_supported",
                    _last_skill_name(trace),
                    metrics,
                    evidence,
                    "re-run support placement and reduce first transport distance",
                    visual_summary,
                )
            force_imbalanced = (
                bool(metrics["payload_support_force_available"])
                and (
                    metrics["payload_support_force_min_robot_n"]
                    < self.min_payload_support_force_per_robot_n
                    or metrics["payload_support_force_balance_ratio"]
                    < self.min_payload_support_force_balance_ratio
                )
            )
            if (
                force_imbalanced
                and not payload_released_on_target
            ):
                evidence["payload_support_force_thresholds"] = {
                    "min_force_per_robot_n": self.min_payload_support_force_per_robot_n,
                    "min_balance_ratio": self.min_payload_support_force_balance_ratio,
                }
                if self.safety.name == "strict":
                    return _fail(
                        "payload_not_supported",
                        _last_skill_name(trace),
                        metrics,
                        evidence,
                        "restore the last balanced checkpoint; both robots must carry measurable palm load before transport",
                        visual_summary,
                    )
                _add_quality_warning(
                    evidence,
                    "support_force_imbalance",
                    "both legal robot supports remain; retain force imbalance for the next decision",
                )
        if metrics["wall_contact_count"] > task.max_wall_contacts:
            if self.safety.wall_contact_is_hard_failure:
                return _fail(
                    "payload_wall_contact",
                    _last_skill_name(trace),
                    metrics,
                    evidence,
                    "insert lateral correction or reduce forward segment length before wall contact",
                    visual_summary,
                )
            _add_quality_warning(
                evidence,
                "payload_wall_contact",
                "contact is undesirable route-quality evidence but does not imply a dropped payload",
            )
        if any(item.status.startswith("failed") for item in trace.skill_calls):
            if monitored_failure is not None:
                return _fail(
                    monitored_failure,
                    _last_action_skill(trace),
                    metrics,
                    evidence,
                    "SONIC transport safety monitor aborted this physical step; restore and evolve its motion parameters",
                    visual_summary,
                )
            return _fail(
                "script_error",
                _last_skill_name(trace),
                metrics,
                evidence,
                "fix failed static skill command invocation",
                visual_summary,
            )
        if monitored_failure is not None:
            return _fail(
                monitored_failure,
                _last_action_skill(trace),
                metrics,
                evidence,
                "SONIC transport safety monitor aborted this physical step; restore and evolve its motion parameters",
                visual_summary,
            )
        action = _last_action_skill(trace)
        if stepwise and action == "release_payload" and not payload_released_on_target:
            return _fail(
                "payload_not_on_target",
                action,
                metrics,
                evidence,
                (
                    "restore the loaded checkpoint; release is accepted only after the "
                    "release event completes with target contact and no floor or wall contact"
                ),
                visual_summary,
            )
        if stepwise and task.require_robots_exit:
            if action in {
                "exit_forward",
                "exit_lateral_left",
                "exit_lateral_right",
            }:
                return _fail(
                    "script_error",
                    action,
                    metrics,
                    evidence,
                    (
                        "the formal post-release lifecycle requires one identical "
                        "synchronized exit_backward transaction while both robots remain crouched"
                    ),
                    visual_summary,
                )
            if action == "exit_backward":
                if not before_robots_crouched or not robots_crouched:
                    return _fail(
                        "script_error",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the released checkpoint; both robots must remain in "
                            "the verified crouch before, during, and after synchronized backward retreat"
                        ),
                        visual_summary,
                    )
                if not lifecycle["crouched_retreat_completed"]:
                    return _fail(
                        "robots_not_clear",
                        action,
                        metrics,
                        evidence,
                        (
                            "the runtime did not observe completion of the synchronized "
                            "crouched backward-retreat phase"
                        ),
                        visual_summary,
                    )
            if action == "recover_stand":
                if not (
                    before_lifecycle["crouched_retreat_completed"]
                    and before_robots_crouched
                    and bool(before_target_metrics["robots_clear_for_stand"])
                ):
                    return _fail(
                        "script_error",
                        action,
                        metrics,
                        evidence,
                        (
                            "recover_stand is legal only after an accepted crouched "
                            "exit_backward checkpoint has cleared both robots from the box "
                            "with enough reserve for stand-recovery drift"
                        ),
                        visual_summary,
                    )
                if not lifecycle["stand_recovered"] or not robots_upright:
                    return _fail(
                        "robot_fall",
                        action,
                        metrics,
                        evidence,
                        (
                            "the final synchronized stand recovery did not leave both "
                            "robot pelvises upright"
                        ),
                        visual_summary,
                    )
                if not bool(metrics["robots_clear_after_place"]):
                    return _fail(
                        "robots_not_clear",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the latest crouched checkpoint; stand recovery moved "
                            "a robot inside the final target-clearance boundary, so add one "
                            "synchronized crouched backward segment before standing again"
                        ),
                        visual_summary,
                    )
        if (
            stepwise
            and not task_complete
            and action not in CONTINUOUS_TURN_ACTIONS
        ):
            endpoint_alignment = _endpoint_longitudinal_alignment_evidence(metrics)
            evidence["endpoint_longitudinal_alignment"] = endpoint_alignment
            if endpoint_alignment["required"]:
                _add_quality_warning(
                    evidence,
                    "endpoint_longitudinal_stagger",
                    (
                        "checkpoint remains physically acceptable; the next transaction "
                        "must move only the lagging robot a short body-forward distance"
                    ),
                )
        if action == PAYLOAD_INWARD_RECOVERY_ACTION:
            recovery_complete, recovery_reason, recovery_evidence = (
                _payload_inward_recovery_completion(state)
            )
            evidence["payload_inward_recovery"] = recovery_evidence
            if not recovery_complete:
                return _fail(
                    "script_error",
                    action,
                    metrics,
                    evidence,
                    (
                        "do not accept this checkpoint; the synchronized motor-side "
                        f"payload recovery did not complete: {recovery_reason}"
                    ),
                    visual_summary,
                )
        if stepwise and action in LINEAR_PROGRESS_ACTIONS:
            verification_contract = (
                trace.verification_contract
                if isinstance(trace.verification_contract, dict)
                else {}
            )
            sync_repair_probe_active = bool(
                verification_contract.get("kind")
                == FORMATION_SYNC_REPAIR_PROBE_KIND
                and verification_contract.get("primitive")
                == COORDINATED_LINEAR_TRANSPORT
                and action == COORDINATED_LINEAR_TRANSPORT
            )
            if sync_repair_probe_active:
                requested_distance = _requested_distance(trace, action)
                required_distance = max(
                    0.0,
                    _float(
                        verification_contract.get("required_distance_m"),
                        1.0,
                    ),
                )
                minimum_progress = max(
                    MIN_MEANINGFUL_TRANSLATION_M,
                    _float(
                        verification_contract.get("minimum_actual_progress_m"),
                        FORMATION_SYNC_REPAIR_MIN_PROGRESS_M,
                    ),
                )
                maximum_mismatch = max(
                    0.0,
                    _float(
                        verification_contract.get(
                            "maximum_normalized_displacement_mismatch"
                        ),
                        FORMATION_SYNC_REPAIR_MAX_NORMALIZED_MISMATCH,
                    ),
                )
                front_displacement = metrics["front_step_displacement_m"]
                back_displacement = metrics["back_step_displacement_m"]
                payload_displacement = metrics["payload_step_displacement_xy_m"]
                displacement_scale = max(
                    front_displacement,
                    back_displacement,
                    1.0e-9,
                )
                normalized_mismatch = (
                    abs(front_displacement - back_displacement)
                    / displacement_scale
                )
                distance_contract_satisfied = bool(
                    abs(requested_distance - required_distance) <= 1.0e-9
                )
                meaningful_progress = bool(
                    min(front_displacement, back_displacement)
                    >= minimum_progress
                    and payload_displacement >= minimum_progress
                )
                mismatch_bounded = bool(
                    normalized_mismatch <= maximum_mismatch + 1.0e-9
                )
                no_wall_contact = metrics["wall_contact_count"] <= 0.0
                target_placement_override = bool(
                    _is_final_route_approach_phase(route_phase_id)
                    and metrics["payload_target_xy_error_m"]
                    <= task.target_xy_tolerance_m + 1.0e-9
                )
                probe_evidence = {
                    "active": True,
                    "passed": bool(
                        distance_contract_satisfied
                        and meaningful_progress
                        and (mismatch_bounded or target_placement_override)
                        and no_wall_contact
                    ),
                    "required_distance_m": required_distance,
                    "requested_distance_m": requested_distance,
                    "distance_contract_satisfied": distance_contract_satisfied,
                    "minimum_actual_progress_m": minimum_progress,
                    "meaningful_actual_progress": meaningful_progress,
                    "front_displacement_m": front_displacement,
                    "back_displacement_m": back_displacement,
                    "payload_displacement_m": payload_displacement,
                    "normalized_displacement_mismatch": normalized_mismatch,
                    "maximum_normalized_displacement_mismatch": maximum_mismatch,
                    "mismatch_bounded": mismatch_bounded,
                    "target_placement_override": target_placement_override,
                    "no_wall_contact": no_wall_contact,
                    "clearance_rule": (
                        "Only a physically accepted probe with meaningful loaded "
                        "motion and bounded front/back response clears the repair gate. "
                        "At the final placement tolerance, legal supported arrival "
                        "takes precedence over residual endpoint mismatch."
                    ),
                }
                evidence["formation_sync_repair_probe"] = probe_evidence
                metrics["formation_sync_normalized_displacement_mismatch"] = (
                    normalized_mismatch
                )
                if not distance_contract_satisfied:
                    return _fail(
                        "script_error",
                        action,
                        metrics,
                        evidence,
                        (
                            "reject before reopening long transport; the formation-sync "
                            f"verification transaction must request exactly "
                            f"{required_distance:g}m"
                        ),
                        visual_summary,
                    )
                if not meaningful_progress:
                    return _fail(
                        "no_payload_progress",
                        action,
                        metrics,
                        evidence,
                        (
                            "the formation-sync probe did not move both robots and the "
                            "supported payload far enough to verify the revised controller"
                        ),
                        visual_summary,
                    )
                if not mismatch_bounded and not target_placement_override:
                    return _fail(
                        "formation_desync",
                        action,
                        metrics,
                        evidence,
                        (
                            "keep the formation-sync repair gate active; the measured "
                            "front/back displacement mismatch remains above the normalized "
                            f"{maximum_mismatch:.2f} limit"
                        ),
                        visual_summary,
                    )
                if not no_wall_contact:
                    return _fail(
                        "payload_wall_contact",
                        action,
                        metrics,
                        evidence,
                        (
                            "the synchronization probe contacted a wall; restore the "
                            "checkpoint and verify the controller in a clear loaded segment"
                        ),
                        visual_summary,
                    )
            formation_skew_deg = metrics["robot_pair_formation_skew_deg"]
            formation_skew_diagnostic = (
                formation_skew_deg > MAX_STRAIGHT_PATH_ERROR_DEG + 1.0e-9
            )
            formation_skew_severe = (
                formation_skew_deg > MAX_SEVERE_LINEAR_DEVIATION_DEG + 1.0e-9
            )
            formation_skew_before_deg = metrics[
                "robot_pair_formation_skew_before_deg"
            ]
            formation_skew_improvement_deg = metrics[
                "robot_pair_formation_skew_improvement_deg"
            ]
            improving_inherited_severe_skew = bool(
                self.safety.name != "strict"
                and formation_skew_severe
                and formation_skew_before_deg
                > MAX_SEVERE_LINEAR_DEVIATION_DEG + 1.0e-9
                and formation_skew_improvement_deg >= 0.5
            )
            formation_skew_hard_failure = bool(
                (formation_skew_severe and not improving_inherited_severe_skew)
                or (self.safety.name == "strict" and formation_skew_diagnostic)
            )
            evidence["linear_formation_geometry"] = {
                "violated": formation_skew_diagnostic,
                "hard_failure": formation_skew_hard_failure,
                "robot_pair_formation_skew_deg": formation_skew_deg,
                "robot_pair_formation_skew_before_deg": formation_skew_before_deg,
                "robot_pair_formation_skew_improvement_deg": (
                    formation_skew_improvement_deg
                ),
                "improving_inherited_severe_skew": improving_inherited_severe_skew,
                "robot_pair_longitudinal_stagger_m": metrics[
                    "robot_pair_longitudinal_stagger_m"
                ],
                "robot_pair_lateral_span_m": metrics["robot_pair_lateral_span_m"],
                "diagnostic_limit_deg": MAX_STRAIGHT_PATH_ERROR_DEG,
                "severe_limit_deg": MAX_SEVERE_LINEAR_DEVIATION_DEG,
                "contract": (
                    "loaded linear translation should preserve an "
                    "approximately side-by-side formation; moderate skew is retained "
                    "as evolution evidence while legal dual support remains, and a "
                    "task-profile rollout may monotonically repair skew inherited "
                    "from its trusted parent"
                ),
            }
            if formation_skew_hard_failure:
                return _fail(
                    "formation_desync",
                    action,
                    metrics,
                    evidence,
                    (
                        "restore the last side-by-side checkpoint; do not continue linear "
                        "transport from a staggered formation. Use a turn only for intentional "
                        "inner/outer geometry, then realign before the next straight segment"
                    ),
                    visual_summary,
                )
            if formation_skew_diagnostic:
                _add_quality_warning(
                    evidence,
                    "linear_formation_skew",
                    "loaded formation skew is visible but both robots still support the payload",
                )
            if bool(straightness_evidence.get("violated", False)):
                severe_straightness = max(
                    abs(metrics["straight_heading_error_deg"]),
                    metrics["straight_robot_heading_split_deg"],
                    abs(metrics["straight_path_angle_error_deg"]),
                ) > MAX_SEVERE_LINEAR_DEVIATION_DEG + 1.0e-9
                straightness_hard_failure = bool(
                    severe_straightness or self.safety.name == "strict"
                )
                straightness_evidence["hard_failure"] = straightness_hard_failure
                straightness_evidence["severe_deviation"] = severe_straightness
                straightness_evidence["severe_limit_deg"] = (
                    MAX_SEVERE_LINEAR_DEVIATION_DEG
                )
                correction_method = str(
                    straightness_evidence.get(
                        "correction_method",
                        COORDINATED_PIVOT,
                    )
                )
                correction_instruction = (
                    "one asynchronous coordinated_alignment(axis='yaw') using the "
                    "reported per-robot deltas"
                    if correction_method == COORDINATED_ALIGNMENT
                    else (
                        "one asynchronous coordinated_pivot in the reported direction, "
                        "with a stationary inner robot and a forward-moving outer robot"
                    )
                )
                if straightness_hard_failure:
                    return _fail(
                        "straightness_regression",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous straight checkpoint, then use exactly "
                            f"{correction_instruction} before any further forward or "
                            "lateral translation"
                        ),
                        visual_summary,
                    )
                _add_quality_warning(
                    evidence,
                    "straightness_regression",
                    (
                        "moderate loaded-heading or path drift remains below the task "
                        "profile's severe boundary; retain it as evidence for the next "
                        "alignment decision while legal dual support continues"
                    ),
                )
            support_center_error = metrics["payload_support_center_error_m"]
            if (
                bool(support_center_evidence["available"])
                and support_center_error > self.max_payload_support_center_error_m
            ):
                evidence["payload_support_center_error_threshold_m"] = (
                    self.max_payload_support_center_error_m
                )
                if self.safety.name == "strict":
                    return _fail(
                        "payload_slip",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the last physically centered checkpoint and use the verified "
                            "equal-speed, smooth-ramp SONIC envelope; cumulative payload offset "
                            "from the live four-palm target must stay bounded"
                        ),
                        visual_summary,
                    )
                _add_quality_warning(
                    evidence,
                    "payload_support_center_offset",
                    "payload remains carried; use this offset to select a corrective next skill",
                )
            vector_desync = metrics["front_back_step_vector_desync_m"]
            yaw_drift = metrics["payload_step_yaw_drift_deg"]
            payload_slip = metrics["payload_formation_slip_m"]
            desync_has_load_risk = (
                payload_slip > 0.5 * self.max_payload_formation_slip_m
                and (
                    bool(support_center_evidence["available"])
                    and support_center_error
                    > 0.8 * self.max_payload_support_center_error_m
                )
            )
            if (
                yaw_drift > self.max_payload_step_yaw_drift_deg
                or (
                    vector_desync > self.max_robot_step_vector_desync_m
                    and desync_has_load_risk
                )
            ):
                evidence["formation_synchronization_thresholds"] = {
                    "max_robot_step_vector_desync_m": self.max_robot_step_vector_desync_m,
                    "max_payload_step_yaw_drift_deg": self.max_payload_step_yaw_drift_deg,
                    "load_risk_required_for_root_desync": True,
                }
                if self.safety.name == "strict":
                    return _fail(
                        "formation_desync",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and recalibrate the front/back SONIC "
                            "executor speeds; do not encode backend compensation in Agent scripts"
                        ),
                        visual_summary,
                    )
                _add_quality_warning(
                    evidence,
                    "formation_desync",
                    "formation mismatch is recoverable while both robots still support the payload",
                )
            if vector_desync > self.max_robot_step_vector_desync_m:
                evidence["formation_synchronization_warning"] = {
                    "front_back_step_vector_desync_m": vector_desync,
                    "payload_formation_slip_m": payload_slip,
                    "payload_support_center_error_m": support_center_error,
                    "reason": (
                        "root-step mismatch retained as rollout evidence because legal palm "
                        "support and payload coupling stayed within their physical limits"
                    ),
                }
            payload_translation = metrics["payload_step_displacement_xy_m"]
            tracking_ratio = metrics["robot_payload_translation_tracking_ratio"]
            if (
                payload_slip > self.max_payload_formation_slip_m
                or (
                    payload_translation >= 0.01
                    and tracking_ratio < self.min_robot_payload_translation_ratio
                )
            ):
                evidence["payload_formation_slip_threshold_m"] = (
                    self.max_payload_formation_slip_m
                )
                evidence["min_robot_payload_translation_ratio"] = (
                    self.min_robot_payload_translation_ratio
                )
                if self.safety.name == "strict":
                    return _fail(
                        "payload_slip",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and shorten the transport window; "
                            "the payload center must move with the two-robot formation instead of "
                            "sliding across the palm supports"
                        ),
                        visual_summary,
                    )
                _add_quality_warning(
                    evidence,
                    "payload_formation_slip",
                    "coupling quality degraded but the legal dual support state remains valid",
                )
        if (
            stepwise
            and action in LINEAR_PROGRESS_ACTIONS
            and metrics["payload_step_displacement_xy_m"]
            >= MIN_MEANINGFUL_TRANSLATION_M
            and (
                metrics["payload_target_xy_error_m"]
                + metrics["payload_final_target_progress_m"]
            )
            <= 2.0 * task.target_xy_tolerance_m + 1.0e-9
            and metrics["payload_final_target_progress_m"]
            <= -MIN_MEANINGFUL_TRANSLATION_M
        ):
            target_distance_before_m = (
                metrics["payload_target_xy_error_m"]
                + metrics["payload_final_target_progress_m"]
            )
            evidence["route_direction_verification"] = {
                "decision": "reject_target_capture_regression",
                "failure_scope": "checkpoint_local_semantic_motion_objective",
                "route_phase_id": str(route_phase_id).strip(),
                "target_capture_radius_m": 2.0 * task.target_xy_tolerance_m,
                "payload_target_distance_before_m": target_distance_before_m,
                "payload_target_xy_error_m": metrics[
                    "payload_target_xy_error_m"
                ],
                "payload_step_displacement_xy_m": metrics[
                    "payload_step_displacement_xy_m"
                ],
                "payload_final_target_progress_m": metrics[
                    "payload_final_target_progress_m"
                ],
                "target_progress_used_for_route_decision": True,
                "route_discovery": bool(task.route_discovery),
                "reason": (
                    "the payload was already inside the final target capture "
                    "region, so a meaningful target-away translation cannot be "
                    "accepted as progress under a stale route-phase label"
                ),
            }
            return _fail(
                "route_regression",
                action,
                metrics,
                evidence,
                (
                    "restore the previous trusted checkpoint and use a target-local "
                    "translation from the live final-target body-frame projection; "
                    "do not continue a room-scale passage action near the target"
                ),
                visual_summary,
            )
        if (
            stepwise
            and _is_final_route_approach_phase(route_phase_id)
            and action in LINEAR_PROGRESS_ACTIONS
            and metrics["payload_step_displacement_xy_m"]
            >= MIN_MEANINGFUL_TRANSLATION_M
            and metrics["payload_final_target_progress_m"]
            <= -MIN_MEANINGFUL_TRANSLATION_M
        ):
            evidence["route_direction_verification"] = {
                "decision": "reject_final_approach_target_regression",
                "failure_scope": "checkpoint_local_semantic_motion_objective",
                "route_phase_id": str(route_phase_id).strip(),
                "payload_step_displacement_xy_m": metrics[
                    "payload_step_displacement_xy_m"
                ],
                "payload_active_stage_progress_m": metrics[
                    "payload_target_progress_m"
                ],
                "payload_final_target_progress_m": metrics[
                    "payload_final_target_progress_m"
                ],
                "target_progress_used_for_route_decision": True,
                "route_discovery": bool(task.route_discovery),
                "reason": (
                    "the committed final-straight phase has no remaining detour "
                    "subtask, so meaningful target-away translation is regression"
                ),
            }
            return _fail(
                "route_regression",
                action,
                metrics,
                evidence,
                (
                    "restore the previous trusted checkpoint and change the final-"
                    "approach semantic direction or heading; do not tune the magnitude "
                    "of a target-away translation"
                ),
                visual_summary,
            )
        route_regression = _route_regression_evidence(
            visual_report,
            action=action,
            metrics=metrics,
            use_target_progress=not task.route_discovery,
            local_navigation_transition=(
                evaluate_local_navigation_transition(
                    local_navigation_contract,
                    before_state,
                    state,
                    requested_turn_direction=_requested_turn_direction(
                        trace,
                        action,
                    ),
                )
                if action in CONTINUOUS_TURN_ACTIONS
                and local_navigation_contract
                else None
            ),
        )
        if task.route_discovery and stepwise and route_regression:
            evidence["route_direction_verification"] = route_regression
            return _fail(
                "route_regression",
                action,
                metrics,
                evidence,
                (
                    "restore the previous trusted checkpoint and change the semantic motion "
                    "direction or geometry-changing coordination primitive; do not retry this "
                    "visually rejected direction by tuning magnitude"
                ),
                visual_summary,
            )
        visual_unavailable = bool(
            visual_report
            and (
                visual_report.status != "ok"
                or (
                    isinstance(visual_report.evidence, dict)
                    and visual_report.evidence.get("scene_visible") is False
                )
            )
        )
        if visual_report and visual_report.blocks_success:
            evidence["visual_advisory_warning"] = {
                "reported_failure_type": visual_report.failure_type,
                "confidence": visual_report.confidence,
                "reason": (
                    "fresh MuJoCo physics passed; model-only safety or task-progress "
                    "labels remain advisory unless the route-regression contract is "
                    "independently corroborated"
                ),
            }
        if stepwise:
            if (
                self.safety.name == "strict"
                and action == "place_payload_on_arms"
                and metrics["payload_tilt_deg"] > 5.0
            ):
                return _fail(
                    "payload_unstable",
                    action,
                    metrics,
                    evidence,
                    "recalibrate or resynchronize the carry support before transport",
                    visual_summary,
                )
            if action in LINEAR_PROGRESS_ACTIONS and not task_complete:
                requested_distance = _requested_distance(trace, action)
                required_displacement = max(0.01, min(0.05, 0.10 * requested_distance))
                metrics["required_step_displacement_m"] = required_displacement
                if (
                    task.route_discovery
                    and metrics["payload_step_displacement_xy_m"] < required_displacement
                ):
                    at_verified_distance_ceiling = bool(
                        requested_distance
                        >= VERIFIED_LOADED_LINEAR_MAX_M - 1e-6
                    )
                    if at_verified_distance_ceiling:
                        evidence["linear_no_progress_at_verified_ceiling"] = {
                            "requested_distance_m": requested_distance,
                            "verified_distance_ceiling_m": (
                                VERIFIED_LOADED_LINEAR_MAX_M
                            ),
                            "measured_payload_displacement_m": metrics[
                                "payload_step_displacement_xy_m"
                            ],
                            "required_step_displacement_m": required_displacement,
                            "reason": (
                                "the largest verified linear transaction produced "
                                "negligible motion, so magnitude tuning cannot repair "
                                "the current direction or maneuver"
                            ),
                        }
                    return _fail(
                        "no_payload_progress",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and change the semantic "
                            "direction or maneuver family"
                            if at_verified_distance_ceiling
                            else (
                                "restore the previous checkpoint and evolve one bounded "
                                "execution magnitude; the selected visual-route experiment "
                                "did not produce measurable payload motion"
                            )
                        ),
                        visual_summary,
                    )
                if (
                    not task.route_discovery
                    and metrics["payload_step_displacement_xy_m"] >= required_displacement
                    and metrics["payload_target_progress_m"] <= -required_displacement
                ):
                    return _fail(
                        "goal_regression",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint; this body-frame action produced "
                            "measurable world motion away from the active goal. Select a different "
                            "direction or phase coordination from the measured displacement, robot "
                            "yaw, active-goal vector, camera geometry, and collision evidence"
                        ),
                        visual_summary,
                    )
                if (
                    not task.route_discovery
                    and metrics["payload_target_progress_m"] < required_displacement
                ):
                    return _fail(
                        "no_payload_progress",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and evolve this segment; "
                            "payload-to-target distance must decrease by at least "
                            f"{required_displacement:.3f}m"
                        ),
                        visual_summary,
                    )
            if action in CONTINUOUS_TURN_ACTIONS and not task_complete:
                active_turn_episode = (
                    turn_episode if isinstance(turn_episode, dict) else {}
                )
                is_heading_correction = bool(
                    active_turn_episode.get("active", False)
                    and str(active_turn_episode.get("purpose", ""))
                    == "heading_correction"
                )
                requested_turn_direction = _requested_turn_direction(trace, action)
                requested_turn_sign = {
                    "clockwise": -1.0,
                    "counterclockwise": 1.0,
                }.get(requested_turn_direction, 0.0)
                local_navigation_transition = (
                    evaluate_local_navigation_transition(
                        local_navigation_contract,
                        before_state,
                        state,
                        requested_turn_direction=requested_turn_direction,
                    )
                    if local_navigation_contract
                    else {}
                )
                if local_navigation_transition:
                    evidence["local_navigation_transition"] = (
                        local_navigation_transition
                    )
                metrics["requested_turn_sign"] = requested_turn_sign
                signed_turn_response = {
                    "requested_turn_direction": requested_turn_direction,
                    "requested_turn_sign": requested_turn_sign,
                    "payload_signed_yaw_change_deg": metrics[
                        "payload_step_signed_yaw_change_deg"
                    ],
                    "team_signed_yaw_change_deg": metrics[
                        "team_step_signed_yaw_change_deg"
                    ],
                    "front_signed_yaw_change_deg": metrics[
                        "front_step_signed_yaw_change_deg"
                    ],
                    "back_signed_yaw_change_deg": metrics[
                        "back_step_signed_yaw_change_deg"
                    ],
                    "minimum_material_yaw_deg": MIN_MEANINGFUL_YAW_DEG,
                    "convention": "+yaw=counterclockwise, -yaw=clockwise",
                }
                opposite_robot_ids = [
                    robot_id
                    for robot_id in ("front", "back")
                    if abs(metrics[f"{robot_id}_step_signed_yaw_change_deg"])
                    >= MIN_MEANINGFUL_YAW_DEG
                    and requested_turn_sign
                    * metrics[f"{robot_id}_step_signed_yaw_change_deg"]
                    < 0.0
                ]
                payload_opposed_request = bool(
                    requested_turn_sign
                    and abs(metrics["payload_step_signed_yaw_change_deg"])
                    >= MIN_MEANINGFUL_YAW_DEG
                    and requested_turn_sign
                    * metrics["payload_step_signed_yaw_change_deg"]
                    < 0.0
                )
                team_opposed_request = bool(
                    requested_turn_sign
                    and abs(metrics["team_step_signed_yaw_change_deg"])
                    >= MIN_MEANINGFUL_YAW_DEG
                    and requested_turn_sign
                    * metrics["team_step_signed_yaw_change_deg"]
                    < 0.0
                )
                signed_turn_response.update(
                    {
                        "opposite_robot_ids": opposite_robot_ids,
                        "payload_opposed_request": payload_opposed_request,
                        "team_opposed_request": team_opposed_request,
                    }
                )
                evidence["signed_turn_verification"] = signed_turn_response
                if (
                    not requested_turn_sign
                    or opposite_robot_ids
                    or (
                        payload_opposed_request
                        and not is_heading_correction
                    )
                    or team_opposed_request
                ):
                    return _fail(
                        "alignment_regression",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint; the measured signed yaw must "
                            "match the requested clockwise/counterclockwise direction for the "
                            "payload and both robot roots before this turn can be accepted"
                        ),
                        visual_summary,
                    )
                meaningful_payload_turn = (
                    metrics["payload_step_yaw_drift_deg"]
                    >= MIN_MEANINGFUL_YAW_DEG
                )
                if is_heading_correction:
                    correction_angle_deg = abs(
                        _float(
                            active_turn_episode.get("correction_angle_deg"),
                            0.0,
                        )
                    )
                    required_team_turn_deg = min(
                        MIN_MEANINGFUL_YAW_DEG,
                        max(0.5, 0.5 * correction_angle_deg),
                    )
                    signed_robot_turns = {
                        robot_id: (
                            requested_turn_sign
                            * metrics[
                                f"{robot_id}_step_signed_yaw_change_deg"
                            ]
                        )
                        for robot_id in ("front", "back")
                    }
                    meaningful_robot_turn = bool(
                        requested_turn_sign
                        and min(signed_robot_turns.values())
                        >= required_team_turn_deg
                    )
                    evidence["heading_correction_verification"] = {
                        "active": True,
                        "requested_correction_angle_deg": correction_angle_deg,
                        "required_per_robot_yaw_deg": required_team_turn_deg,
                        "signed_robot_yaw_deg": signed_robot_turns,
                        "payload_yaw_is_diagnostic_only": True,
                        "reason": (
                            "straight-heading correction aligns the two locomotion "
                            "roots; a freely supported payload need not rotate by the "
                            "same small angle"
                        ),
                    }
                else:
                    meaningful_robot_turn = (
                        metrics["max_robot_step_yaw_change_deg"]
                        >= MIN_MEANINGFUL_YAW_DEG
                    )
                meaningful_arc = metrics["payload_step_displacement_xy_m"] >= 0.02
                if meaningful_payload_turn and not meaningful_robot_turn:
                    return _fail(
                        "payload_slip",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint; payload yaw changed without "
                            "corresponding robot yaw, so the box rotated across its supports"
                        ),
                        visual_summary,
                    )
                turn_regression_limit = max(
                    0.05,
                    0.25 * metrics["payload_step_displacement_xy_m"],
                )
                target_progress = metrics["payload_target_progress_m"]
                heading_alignment_improvement = metrics[
                    "team_forward_alignment_active_goal_improvement"
                ]
                payload_axis_improvement = metrics[
                    "payload_long_axis_alignment_active_goal_improvement"
                ]
                turn_progress = {
                    "target_progress_m": target_progress,
                    "max_allowed_goal_regression_m": turn_regression_limit,
                    "team_forward_alignment_improvement": (
                        heading_alignment_improvement
                    ),
                    "payload_axis_alignment_improvement": payload_axis_improvement,
                    "minimum_alignment_improvement": MIN_TURN_ALIGNMENT_IMPROVEMENT,
                }
                evidence["turn_progress_verification"] = turn_progress
                semantic_turn_improved = bool(
                    local_navigation_transition.get(
                        "semantic_heading_improved",
                        False,
                    )
                    if local_navigation_transition.get("available", False)
                    else meaningful_robot_turn
                    if is_heading_correction
                    else (
                        meaningful_payload_turn and meaningful_robot_turn
                        if task.route_discovery
                        else (
                        target_progress >= MIN_MEANINGFUL_TRANSLATION_M
                        or heading_alignment_improvement
                        >= MIN_TURN_ALIGNMENT_IMPROVEMENT
                        or payload_axis_improvement >= MIN_TURN_ALIGNMENT_IMPROVEMENT
                        )
                    )
                )
                turn_progress["semantic_turn_improved"] = semantic_turn_improved
                if local_navigation_transition.get("available", False):
                    turn_progress["success_reference"] = (
                        "declared_local_heading_error"
                    )
                    turn_progress["target_distance_is_diagnostic_only"] = True
                turn_progress["meaningful_payload_turn"] = meaningful_payload_turn
                turn_progress["meaningful_robot_turn"] = meaningful_robot_turn
                turn_progress["meaningful_arc_translation"] = meaningful_arc
                if (
                    not meaningful_robot_turn
                    or (action == COORDINATED_ARC_TRANSPORT and not meaningful_arc)
                    or (
                        action in STATIONARY_LOADED_TURN_ACTIONS
                        and not meaningful_payload_turn
                        and not is_heading_correction
                    )
                    or (
                        action == COORDINATED_ARC_TRANSPORT
                        and not meaningful_payload_turn
                        and not semantic_turn_improved
                    )
                ):
                    return _fail(
                        "no_payload_progress",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and repair one bounded shared-heading "
                            "turn; a loaded arc must measurably move both robots, while a static "
                            "turn or pivot must also rotate the payload"
                        ),
                        visual_summary,
                    )
                if (
                    not task.route_discovery
                    and not local_navigation_transition.get("available", False)
                    and target_progress <= -turn_regression_limit
                ):
                    return _fail(
                        "goal_regression",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and switch the shared turn direction, "
                            "inner/outer role geometry, or primitive; a continuous turn may use a "
                            "small setup arc, but it must not carry the payload substantially away "
                            "from the active stage goal"
                        ),
                        visual_summary,
                    )
                if (
                    local_navigation_transition.get("available", False)
                    and not semantic_turn_improved
                ):
                    return _fail(
                        "alignment_regression",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and choose a signed "
                            "shared-heading turn that reduces the absolute heading "
                            "error to the Task Orchestrator's declared local reference"
                        ),
                        visual_summary,
                    )
                if (
                    not task.route_discovery
                    and not local_navigation_transition.get("available", False)
                    and not semantic_turn_improved
                ):
                    return _fail(
                        "alignment_regression",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and change the turn objective or role "
                            "geometry; measurable yaw is not useful turn progress unless target "
                            "distance, team forward alignment, or payload-axis alignment improves"
                        ),
                        visual_summary,
                    )
            if action == COORDINATED_ALIGNMENT and not task_complete:
                alignment_axis = _requested_parameter(trace, action, "axis")
                if alignment_axis == "yaw":
                    meaningful_robot_alignment = (
                        metrics["max_robot_step_yaw_change_deg"]
                        >= MIN_MEANINGFUL_YAW_DEG
                    )
                    payload_only_motion = (
                        metrics["payload_step_yaw_drift_deg"]
                        >= MIN_MEANINGFUL_YAW_DEG
                        and not meaningful_robot_alignment
                    )
                else:
                    meaningful_robot_alignment = (
                        max(
                            metrics["front_step_displacement_m"],
                            metrics["back_step_displacement_m"],
                        )
                        >= MIN_MEANINGFUL_TRANSLATION_M
                        and metrics["front_back_step_vector_desync_m"]
                        >= MIN_MEANINGFUL_TRANSLATION_M
                    )
                    payload_only_motion = (
                        metrics["payload_formation_slip_m"]
                        >= MIN_MEANINGFUL_TRANSLATION_M
                        and not meaningful_robot_alignment
                    )
                evidence["alignment_progress_verification"] = {
                    "axis": alignment_axis,
                    "meaningful_robot_alignment": meaningful_robot_alignment,
                    "payload_only_motion": payload_only_motion,
                    "minimum_translation_m": MIN_MEANINGFUL_TRANSLATION_M,
                    "minimum_yaw_deg": MIN_MEANINGFUL_YAW_DEG,
                }
                if payload_only_motion:
                    return _fail(
                        "payload_slip",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint; the payload moved across the "
                            "supports while the requested robot alignment remained below the "
                            "measurable-motion threshold"
                        ),
                        visual_summary,
                    )
                meaningful_alignment = meaningful_robot_alignment
                if not meaningful_alignment:
                    return _fail(
                        "no_payload_progress",
                        action,
                        metrics,
                        evidence,
                        "restore and increase one bounded alignment delta until physics observes it",
                        visual_summary,
                    )
                alignment_improvements = {
                    "robot_heading_disagreement_deg": metrics[
                        "robot_heading_disagreement_improvement_deg"
                    ],
                    "robot_pair_formation_skew_deg": metrics[
                        "robot_pair_formation_skew_improvement_deg"
                    ],
                    "robot_pair_longitudinal_stagger_m": metrics[
                        "robot_pair_longitudinal_stagger_improvement_m"
                    ],
                    "payload_axis_active_goal": metrics[
                        "payload_long_axis_alignment_active_goal_improvement"
                    ],
                    "payload_support_center_error_m": metrics[
                        "payload_support_center_error_improvement_m"
                    ],
                }
                alignment_thresholds = {
                    "robot_heading_disagreement_deg": (
                        MIN_ALIGNMENT_HEADING_IMPROVEMENT_DEG
                    ),
                    "robot_pair_formation_skew_deg": (
                        MIN_ALIGNMENT_FORMATION_SKEW_IMPROVEMENT_DEG
                    ),
                    "robot_pair_longitudinal_stagger_m": (
                        MIN_LONGITUDINAL_STAGGER_IMPROVEMENT_M
                    ),
                    "payload_axis_active_goal": MIN_ALIGNMENT_AXIS_IMPROVEMENT,
                    "payload_support_center_error_m": (
                        MIN_ALIGNMENT_SUPPORT_CENTER_IMPROVEMENT_M
                    ),
                }
                support_center_regressed = bool(
                    support_center_comparison_available
                    and alignment_improvements[
                        "payload_support_center_error_m"
                    ]
                    <= -MIN_ALIGNMENT_SUPPORT_CENTER_IMPROVEMENT_M
                )
                heading_improved = bool(
                    alignment_axis == "yaw"
                    and alignment_improvements[
                        "robot_heading_disagreement_deg"
                    ]
                    >= alignment_thresholds[
                        "robot_heading_disagreement_deg"
                    ]
                )
                lateral_formation_improved = bool(
                    alignment_axis == "lateral"
                    and alignment_improvements[
                        "robot_pair_formation_skew_deg"
                    ]
                    >= alignment_thresholds[
                        "robot_pair_formation_skew_deg"
                    ]
                )
                longitudinal_stagger_improved = bool(
                    alignment_axis == "longitudinal"
                    and alignment_improvements[
                        "robot_pair_longitudinal_stagger_m"
                    ]
                    >= alignment_thresholds[
                        "robot_pair_longitudinal_stagger_m"
                    ]
                )
                verification_contract = (
                    trace.verification_contract
                    if isinstance(trace.verification_contract, dict)
                    else {}
                )
                checkpoint = (
                    before_state.get("checkpoint", {})
                    if isinstance(before_state, dict)
                    else {}
                )
                checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
                contract_checkpoint_id = str(
                    verification_contract.get("scope_checkpoint_id", "")
                ).strip()
                current_checkpoint_id = str(
                    checkpoint.get("checkpoint_id", "")
                ).strip()
                spacing_reserve_contract_active = bool(
                    verification_contract.get("kind")
                    == "formation_spacing_reserve"
                    and verification_contract.get("source")
                    == "verifier_pair_spacing_repair"
                    and verification_contract.get("primitive")
                    == COORDINATED_ALIGNMENT
                    and verification_contract.get("semantic_objective")
                    == alignment_axis
                    and (
                        not contract_checkpoint_id
                        or not current_checkpoint_id
                        or contract_checkpoint_id == current_checkpoint_id
                    )
                )
                spacing_reserve_improved = bool(
                    spacing_reserve_contract_active
                    and alignment_axis in {"longitudinal", "lateral"}
                    and metrics["robot_pair_xy_separation_m"]
                    >= task.min_robot_xy_separation_m
                    and metrics["robot_pair_xy_separation_improvement_m"]
                    >= MIN_MEANINGFUL_TRANSLATION_M
                )
                target_directed_final_alignment = bool(
                    self.safety.name != "strict"
                    and _is_final_route_approach_phase(route_phase_id)
                    and metrics["payload_final_target_progress_m"]
                    >= MIN_MEANINGFUL_TRANSLATION_M
                )
                semantic_alignment_improved = bool(
                    synchronization_evidence.get("available", False)
                    and (
                        spacing_reserve_improved
                        or target_directed_final_alignment
                        or longitudinal_stagger_improved
                        or lateral_formation_improved
                        or heading_improved
                        or (
                            alignment_axis == "yaw"
                            and not task.route_discovery
                            and alignment_improvements["payload_axis_active_goal"]
                            >= alignment_thresholds["payload_axis_active_goal"]
                        )
                    )
                )
                evidence["alignment_progress_verification"].update(
                    {
                        "semantic_alignment_improved": semantic_alignment_improved,
                        "longitudinal_stagger_improved": (
                            longitudinal_stagger_improved
                        ),
                        "lateral_formation_improved": lateral_formation_improved,
                        "heading_improved": heading_improved,
                        "spacing_reserve_contract_active": (
                            spacing_reserve_contract_active
                        ),
                        "spacing_reserve_improved": spacing_reserve_improved,
                        "target_directed_final_alignment": (
                            target_directed_final_alignment
                        ),
                        "support_center_regressed": support_center_regressed,
                        "improvements": alignment_improvements,
                        "improvement_thresholds": alignment_thresholds,
                    }
                )
                if support_center_regressed:
                    if self.safety.name == "strict":
                        return _fail(
                            "payload_slip",
                            action,
                            metrics,
                            evidence,
                            (
                                "restore the previous checkpoint and change the alignment roles "
                                "or axis; strict mode requires every correction to improve the "
                                "live dual-arm support-center error"
                            ),
                            visual_summary,
                        )
                    _add_quality_warning(
                        evidence,
                        "alignment_support_center_regression",
                        (
                            "the alignment retained legal dual support but moved the payload "
                            "slightly farther from the live support center; preserve the "
                            "measured offset as next-step evidence"
                        ),
                    )
                if target_directed_final_alignment and not (
                    longitudinal_stagger_improved
                    or lateral_formation_improved
                    or heading_improved
                    or spacing_reserve_improved
                ):
                    _add_quality_warning(
                        evidence,
                        "final_alignment_formation_tradeoff",
                        (
                            "the bounded alignment moved the supported payload toward the "
                            "physical final target but did not improve its local formation "
                            "diagnostic; preserve both measurements for the next subtask"
                        ),
                    )
                if not semantic_alignment_improved:
                    return _fail(
                        "alignment_regression",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and switch the motion objective "
                            "or coordination primitive; observable robot motion is not progress "
                            "unless it improves the requested axis-specific relation "
                            "(longitudinal stagger, lateral formation skew, or yaw agreement), "
                            "or an active verifier-signed spacing reserve"
                        ),
                        visual_summary,
                    )
            if action in {
                "exit_forward",
                "exit_backward",
                "exit_lateral_left",
                "exit_lateral_right",
            }:
                if not payload_released_on_target:
                    return _fail(
                        "payload_not_on_target",
                        action,
                        metrics,
                        evidence,
                        "restore the released target-supported checkpoint before any unloaded exit",
                        visual_summary,
                    )
                # Exit clearance is cumulative.  The terminal gate below still
                # requires the configured absolute distance, while each safe
                # crouched segment only needs to move the nearest robot farther
                # from the target.  Requiring one segment to exceed an arbitrary
                # centimetre discarded valid progress from the independent SONIC
                # backends and made the lifecycle impossible to accumulate.
                if not task_complete and metrics["robot_exit_progress_m"] <= 0.0:
                    return _fail(
                        "robots_not_clear",
                        action,
                        metrics,
                        evidence,
                        (
                            "restore the previous checkpoint and choose a body-relative exit "
                            "direction that measurably increases both robots' target clearance"
                        ),
                        visual_summary,
                    )
            if action in {"", "stop_all"} and not task_complete:
                return _fail(
                    "no_payload_progress",
                    action,
                    metrics,
                    evidence,
                    "generate one bounded physical task action before stop_all",
                    visual_summary,
                )
            if visual_unavailable and visual_report is not None:
                return _fail(
                    "manual_review_required",
                    _last_action_skill(trace),
                    metrics,
                    evidence,
                    "restore a complete fresh global-camera observation before checkpoint commit",
                    visual_summary,
                )
            return annotate_evaluation_report(EvaluationReport(
                success=True,
                failure_type="none",
                metrics=metrics,
                evidence=evidence,
                recommendations=(
                    "instruction complete" if task_complete else "commit checkpoint and generate the next subtask",
                ),
                visual_summary=visual_summary,
            ))
        if visual_unavailable and visual_report is not None:
            return _fail(
                "manual_review_required",
                _last_action_skill(trace),
                metrics,
                evidence,
                "restore a complete fresh global-camera observation before checkpoint commit",
                visual_summary,
            )
        if metrics["forward_progress_m"] < task.min_forward_progress_m and not payload_on_target:
            return _fail(
                "no_payload_progress",
                _last_skill_name(trace),
                metrics,
                evidence,
                "increase forward distance only after payload contact is stable",
                visual_summary,
            )
        if task.require_payload_on_target and not payload_on_target:
            return _fail(
                "payload_not_on_target",
                _last_skill_name(trace),
                metrics,
                evidence,
                (
                    "place and release the payload so the red target cylinder "
                    "physically supports it without floor, wall, or robot contact"
                ),
                visual_summary,
            )
        if task.require_robots_exit and not bool(metrics["robots_clear_after_place"]):
            return _fail(
                "robots_not_clear",
                _last_skill_name(trace),
                metrics,
                evidence,
                (
                    "after placement, both robots must retreat backward while crouched, "
                    "recover upright, and remain clear of the target: "
                    f"min robot-target distance >= {task.robot_exit_min_distance_from_target_m:.3f}m"
                ),
                visual_summary,
            )
        return annotate_evaluation_report(EvaluationReport(
            success=True,
            failure_type="none",
            metrics=metrics,
            evidence=evidence,
            recommendations=("continue to next subtask/stage",),
            visual_summary=visual_summary,
        ))

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.runtime_state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


def _fail(
    failure_type,
    failed_subtask: str,
    metrics: dict[str, float],
    evidence: dict[str, Any],
    recommendation: str,
    visual_summary: str,
) -> EvaluationReport:
    return annotate_evaluation_report(EvaluationReport(
        success=False,
        failure_type=failure_type,
        failed_subtask=failed_subtask,
        metrics=metrics,
        evidence=evidence,
        recommendations=(recommendation,),
        visual_summary=visual_summary,
    ))


def _add_quality_warning(
    evidence: dict[str, Any],
    warning_type: str,
    reason: str,
) -> None:
    """Retain recoverable rollout quality evidence without forcing rollback."""

    warnings = evidence.setdefault("physical_quality_warnings", [])
    if not isinstance(warnings, list):
        warnings = []
        evidence["physical_quality_warnings"] = warnings
    item = {"type": str(warning_type), "reason": str(reason)}
    if item not in warnings:
        warnings.append(item)


def _payload_inward_recovery_completion(
    state: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    """Verify the physical and runtime-event contract for arm-only recovery."""

    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    lock = state.get("upper_body_pose_lock", {}) if isinstance(state, dict) else {}
    lock = lock if isinstance(lock, dict) else {}
    last_event = lock.get("last_event", {})
    last_event = last_event if isinstance(last_event, dict) else {}
    supporting_robots = sorted(
        str(robot_id) for robot_id in contacts.get("support_robots", ())
    )
    evidence = {
        "pose_name": lock.get("pose_name", ""),
        "target_pose_name": lock.get("target_pose_name", ""),
        "transition_active": bool(lock.get("transition_active", False)),
        "last_event": last_event,
        "supporting_robots": supporting_robots,
        "floor_contact": bool(contacts.get("floor", False)),
        "body_forward_support_offset_m": payload.get(
            "support_center_signed_longitudinal_offset_m"
        ),
        "waist_and_legs_unchanged": bool(
            lock.get("waist_and_legs_unchanged_by_runtime_events", False)
        ),
    }
    checks = (
        (bool(lock.get("configured", False)), "upper-body pose lock is unavailable"),
        (
            str(lock.get("pose_name", "")) == PAYLOAD_RECOVERY_POSE_NAME,
            f"completed pose is not {PAYLOAD_RECOVERY_POSE_NAME}",
        ),
        (
            str(lock.get("target_pose_name", "")) == PAYLOAD_RECOVERY_POSE_NAME,
            f"target pose is not {PAYLOAD_RECOVERY_POSE_NAME}",
        ),
        (not bool(lock.get("transition_active", False)), "arm transition is still active"),
        (
            str(last_event.get("event", "")) == "recover_payload_inward",
            "Term1 did not report the recovery event",
        ),
        (
            bool(last_event.get("success", False))
            and bool(last_event.get("completed", False)),
            str(last_event.get("error", "")) or "recovery event is incomplete",
        ),
        (
            {"front", "back"}.issubset(supporting_robots),
            "payload no longer has legal front/back support",
        ),
        (not bool(contacts.get("floor", False)), "payload is touching the floor"),
        (
            bool(lock.get("waist_and_legs_unchanged_by_runtime_events", False)),
            "runtime cannot attest that waist and legs were excluded",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason, evidence
    return True, "", evidence


def _last_skill_name(trace: ExecutionTrace) -> str:
    if not trace.skill_calls:
        return ""
    return _canonical_action_skill_name(trace.skill_calls[-1].skill_name)


def _last_action_skill(trace: ExecutionTrace) -> str:
    for item in reversed(trace.skill_calls):
        if item.skill_name != "stop_all":
            return _canonical_action_skill_name(item.skill_name)
    return "stop_all" if trace.skill_calls else ""


def _canonical_action_skill_name(skill_name: str) -> str:
    return {
        "carry_recover_stand": "recover_stand",
    }.get(skill_name, skill_name)


def _requested_distance(trace: ExecutionTrace, action: str) -> float:
    for item in reversed(trace.skill_calls):
        if item.skill_name == action:
            direct = abs(_float(item.parameters.get("distance_m"), 0.0))
            if direct > 0.0:
                return direct
            actions = item.parameters.get("actions", {})
            if isinstance(actions, dict):
                distances = []
                for profile in actions.values():
                    if not isinstance(profile, dict):
                        continue
                    parameters = profile.get("parameters", {})
                    if isinstance(parameters, dict):
                        distances.append(abs(_float(parameters.get("distance_m"), 0.0)))
                if distances:
                    return max(distances)
    return 0.0


def _requested_parameter(
    trace: ExecutionTrace,
    action: str,
    name: str,
) -> str:
    values: list[str] = []
    for item in reversed(trace.skill_calls):
        if item.skill_name != action:
            continue
        if name in item.parameters:
            values.append(str(item.parameters[name]))
        actions = item.parameters.get("actions", {})
        if isinstance(actions, dict):
            for profile in actions.values():
                if not isinstance(profile, dict):
                    continue
                parameters = profile.get("parameters", {})
                if isinstance(parameters, dict) and name in parameters:
                    values.append(str(parameters[name]))
        break
    unique = tuple(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else ""


def _requested_turn_direction(trace: ExecutionTrace, action: str) -> str:
    """Return one semantic turn sign across static and coordination APIs."""

    static_direction = {
        "turn_clockwise": "clockwise",
        "turn_counterclockwise": "counterclockwise",
    }.get(action, "")
    if static_direction:
        return static_direction
    return _requested_parameter(trace, action, "turn_direction").strip().lower()


def _monitored_physical_failure(trace: ExecutionTrace) -> str | None:
    text = "\n".join(
        [
            trace.error,
            *[item.stdout for item in trace.skill_calls],
            *[item.stderr for item in trace.skill_calls],
        ]
    ).lower()
    markers = (
        ("robot_fall", "robot_fall"),
        ("payload_floor_contact", "payload_drop"),
        ("payload_below_safety_height", "payload_drop"),
        ("floor=true", "payload_drop"),
        ("dual_payload_support_lost", "payload_not_supported"),
        ("payload_support_force_imbalanced", "payload_not_supported"),
        ("continuous physical support check", "payload_not_supported"),
        ("payload_hard_tilt_exceeded", "payload_unstable"),
        ("payload_tilt_exceeded", "payload_unstable"),
        ("payload_top_face_inverted", "payload_unstable"),
        ("payload_support_center_error_exceeded", "payload_slip"),
        ("payload_support_center_error_unavailable", "payload_slip"),
        ("payload_slip", "payload_slip"),
        ("payload_wall_contact", "payload_wall_contact"),
        ("payload_not_active", "payload_not_active"),
        ("formation_sync_hard_limit", "formation_desync"),
        ("formation_sync_settle_hard_limit", "formation_desync"),
        ("robot_pair_too_close", "formation_desync"),
        ("robot_pair_too_far", "formation_desync"),
        ("robot_pair_separation_unavailable", "runtime_state_missing"),
        ("runtime_state_missing_or_stale", "runtime_state_missing"),
        ("payload-on-target gate did not pass", "payload_not_on_target"),
        ("payload release gate did not pass", "payload_not_on_target"),
    )
    for marker, failure_type in markers:
        if marker in text:
            return failure_type
    return None


def _payload_step_displacement(before: dict[str, Any] | None, after: dict[str, Any]) -> float:
    if not before:
        return 0.0
    before_payload = before.get("payload", {})
    after_payload = after.get("payload", {})
    if not isinstance(before_payload, dict) or not isinstance(after_payload, dict):
        return 0.0
    dx = _float(after_payload.get("x"), 0.0) - _float(before_payload.get("x"), 0.0)
    dy = _float(after_payload.get("y"), 0.0) - _float(before_payload.get("y"), 0.0)
    dz = _float(after_payload.get("z"), 0.0) - _float(before_payload.get("z"), 0.0)
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _payload_target_progress(
    task: BoxCarryTask,
    before: dict[str, Any] | None,
    after: dict[str, Any],
) -> float:
    """Positive only when the payload gets closer to the active stage goal."""

    if not before:
        return 0.0
    before_payload = before.get("payload", {})
    after_payload = after.get("payload", {})
    if not isinstance(before_payload, dict) or not isinstance(after_payload, dict):
        return 0.0
    goal_x, goal_y = _active_goal_xy(task, before_payload)
    before_distance = hypot(
        _float(before_payload.get("x"), 0.0) - goal_x,
        _float(before_payload.get("y"), 0.0) - goal_y,
    )
    after_distance = hypot(
        _float(after_payload.get("x"), 0.0) - goal_x,
        _float(after_payload.get("y"), 0.0) - goal_y,
    )
    return float(before_distance - after_distance)


def _payload_final_target_progress(
    task: BoxCarryTask,
    before: dict[str, Any] | None,
    after: dict[str, Any],
) -> float:
    """Positive only when the payload gets closer to the physical final target."""

    if not before:
        return 0.0
    before_payload = before.get("payload", {})
    after_payload = after.get("payload", {})
    if not isinstance(before_payload, dict) or not isinstance(after_payload, dict):
        return 0.0
    before_distance = hypot(
        _float(before_payload.get("x"), 0.0) - task.target_center_x_m,
        _float(before_payload.get("y"), 0.0) - task.target_center_y_m,
    )
    after_distance = hypot(
        _float(after_payload.get("x"), 0.0) - task.target_center_x_m,
        _float(after_payload.get("y"), 0.0) - task.target_center_y_m,
    )
    return float(before_distance - after_distance)


def _payload_support_center_metrics(
    payload: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure absolute payload drift from Term1's dynamic four-palm target."""

    target = payload.get("live_support_target", {})
    if not isinstance(target, dict):
        target = {}
    required_values = (
        payload.get("x"),
        payload.get("y"),
        target.get("x"),
        target.get("y"),
    )
    try:
        payload_x, payload_y, target_x, target_y = (
            float(value) for value in required_values
        )
    except (TypeError, ValueError):
        return {"payload_support_center_error_m": 0.0}, {
            "available": False,
            "reason": "payload_or_live_support_target_xy_missing",
        }

    dx = payload_x - target_x
    dy = payload_y - target_y
    return {"payload_support_center_error_m": hypot(dx, dy)}, {
        "available": True,
        "payload_xy_m": [payload_x, payload_y],
        "support_target_xy_m": [target_x, target_y],
        "offset_xy_m": [dx, dy],
    }


def _payload_support_force_metrics(
    contacts: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Expose whether both palm pairs carry load, not merely touch the box."""

    support_forces = contacts.get("support_forces", {})
    if not isinstance(support_forces, dict) or not support_forces.get("available", False):
        return {
            "payload_support_force_available": 0.0,
            "payload_support_force_front_n": 0.0,
            "payload_support_force_back_n": 0.0,
            "payload_support_force_total_n": 0.0,
            "payload_support_force_min_robot_n": 0.0,
            "payload_support_force_balance_ratio": 0.0,
        }, {
            "available": False,
            "reason": "support_force_telemetry_missing",
        }

    force_by_robot = support_forces.get("normal_force_n_by_robot", {})
    if not isinstance(force_by_robot, dict):
        force_by_robot = {}
    front_force = max(0.0, _float(force_by_robot.get("front"), 0.0))
    back_force = max(0.0, _float(force_by_robot.get("back"), 0.0))
    total_force = max(
        0.0,
        _float(support_forces.get("total_normal_force_n"), front_force + back_force),
    )
    balance_ratio = max(0.0, _float(support_forces.get("balance_ratio"), 0.0))
    return {
        "payload_support_force_available": 1.0,
        "payload_support_force_front_n": front_force,
        "payload_support_force_back_n": back_force,
        "payload_support_force_total_n": total_force,
        "payload_support_force_min_robot_n": min(front_force, back_force),
        "payload_support_force_balance_ratio": balance_ratio,
    }, {
        "available": True,
        "normal_force_n_by_robot": {"front": front_force, "back": back_force},
        "total_normal_force_n": total_force,
        "balance_ratio": balance_ratio,
    }


def _translation_straightness_metrics(
    trace: ExecutionTrace,
    before: dict[str, Any] | None,
    after: dict[str, Any],
    *,
    reference_yaw_rad: float | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Audit one intended translation against a persistent heading axis."""

    empty_metrics = {
        "straight_reference_heading_deg": 0.0,
        "straight_mean_heading_before_deg": 0.0,
        "straight_mean_heading_after_deg": 0.0,
        "straight_heading_error_deg": 0.0,
        "straight_front_heading_error_deg": 0.0,
        "straight_back_heading_error_deg": 0.0,
        "straight_robot_heading_split_deg": 0.0,
        "straight_team_along_track_m": 0.0,
        "straight_team_cross_track_m": 0.0,
        "straight_path_angle_error_deg": 0.0,
    }
    action = _last_action_skill(trace)
    if action not in LINEAR_PROGRESS_ACTIONS:
        return empty_metrics, {
            "available": False,
            "reason": "action_is_not_translation",
        }
    if reference_yaw_rad is None:
        return empty_metrics, {
            "available": False,
            "reason": "persistent_heading_reference_missing",
        }
    if not isinstance(before, dict):
        return empty_metrics, {"available": False, "reason": "before_state_missing"}
    before_robots = before.get("robots", {})
    after_robots = after.get("robots", {})
    if not isinstance(before_robots, dict) or not isinstance(after_robots, dict):
        return empty_metrics, {"available": False, "reason": "robot_state_missing"}
    if not all(
        isinstance(robots.get(robot_id), dict)
        and _robot_has_pose_and_yaw(robots[robot_id])
        for robots in (before_robots, after_robots)
        for robot_id in ("front", "back")
    ):
        return empty_metrics, {
            "available": False,
            "reason": "front_back_pose_or_yaw_missing",
        }

    semantic_direction = _translation_semantic_direction(trace, action)
    if semantic_direction not in {"forward", "backward", "left", "right"}:
        return empty_metrics, {
            "available": False,
            "reason": "translation_semantic_direction_missing",
        }

    reference = float(reference_yaw_rad)
    forward = (cos(reference), sin(reference))
    left = (-forward[1], forward[0])
    expected = {
        "forward": forward,
        "backward": (-forward[0], -forward[1]),
        "left": left,
        "right": (-left[0], -left[1]),
    }[semantic_direction]
    expected_left = (-expected[1], expected[0])

    before_yaws = tuple(
        _robot_yaw_rad(before_robots[robot_id]) for robot_id in ("front", "back")
    )
    after_yaws = tuple(
        _robot_yaw_rad(after_robots[robot_id]) for robot_id in ("front", "back")
    )
    mean_before = atan2(
        sum(sin(value) for value in before_yaws),
        sum(cos(value) for value in before_yaws),
    )
    mean_after = atan2(
        sum(sin(value) for value in after_yaws),
        sum(cos(value) for value in after_yaws),
    )
    heading_error_deg = degrees(_wrapped_angle(mean_after - reference))
    robot_heading_errors_deg = {
        robot_id: degrees(_wrapped_angle(yaw - reference))
        for robot_id, yaw in zip(("front", "back"), after_yaws, strict=True)
    }
    heading_split_deg = abs(
        degrees(_wrapped_angle(after_yaws[1] - after_yaws[0]))
    )

    robot_deltas = []
    for robot_id in ("front", "back"):
        before_x, before_y = _robot_xy(before_robots[robot_id])
        after_x, after_y = _robot_xy(after_robots[robot_id])
        robot_deltas.append((after_x - before_x, after_y - before_y))
    team_delta = (
        0.5 * (robot_deltas[0][0] + robot_deltas[1][0]),
        0.5 * (robot_deltas[0][1] + robot_deltas[1][1]),
    )
    along_track = team_delta[0] * expected[0] + team_delta[1] * expected[1]
    cross_track = team_delta[0] * expected_left[0] + team_delta[1] * expected_left[1]
    path_distance = hypot(*team_delta)
    path_angle_error_deg = (
        degrees(atan2(cross_track, along_track))
        if path_distance >= MIN_STRAIGHT_PATH_AUDIT_M
        else 0.0
    )
    heading_violated = abs(heading_error_deg) > MAX_STRAIGHT_HEADING_ERROR_DEG
    split_violated = heading_split_deg > MAX_STRAIGHT_HEADING_ERROR_DEG
    path_violated = bool(
        path_distance >= MIN_STRAIGHT_PATH_AUDIT_M
        and abs(path_angle_error_deg) > MAX_STRAIGHT_PATH_ERROR_DEG
    )
    violated = heading_violated or split_violated or path_violated

    correction_error_deg = (
        heading_error_deg
        if abs(heading_error_deg) > 0.25
        else path_angle_error_deg
    )
    correction_direction = (
        "counterclockwise"
        if correction_error_deg < 0.0
        else "clockwise"
        if correction_error_deg > 0.0
        else "none"
    )
    correction_angle_deg = min(10.0, max(2.0, abs(correction_error_deg)))
    alignment_robot_id = max(
        robot_heading_errors_deg,
        key=lambda robot_id: abs(robot_heading_errors_deg[robot_id]),
    )
    alignment_error_deg = robot_heading_errors_deg[alignment_robot_id]
    alignment_delta_deg = (
        min(10.0, max(2.0, abs(alignment_error_deg)))
        * (-1.0 if alignment_error_deg > 0.0 else 1.0)
        if abs(alignment_error_deg) > 0.25
        else 0.0
    )
    correction_method = (
        COORDINATED_ALIGNMENT
        if split_violated and not heading_violated
        else COORDINATED_PIVOT
    )
    metrics = {
        "straight_reference_heading_deg": degrees(reference),
        "straight_mean_heading_before_deg": degrees(mean_before),
        "straight_mean_heading_after_deg": degrees(mean_after),
        "straight_heading_error_deg": heading_error_deg,
        "straight_front_heading_error_deg": robot_heading_errors_deg["front"],
        "straight_back_heading_error_deg": robot_heading_errors_deg["back"],
        "straight_robot_heading_split_deg": heading_split_deg,
        "straight_team_along_track_m": along_track,
        "straight_team_cross_track_m": cross_track,
        "straight_path_angle_error_deg": path_angle_error_deg,
    }
    return metrics, {
        "available": True,
        "violated": violated,
        "semantic_direction": semantic_direction,
        "expected_world_direction_xy": [expected[0], expected[1]],
        "heading_error_exceeded": heading_violated,
        "robot_heading_split_exceeded": split_violated,
        "path_angle_error_exceeded": path_violated,
        "limits": {
            "max_heading_or_split_error_deg": MAX_STRAIGHT_HEADING_ERROR_DEG,
            "max_path_angle_error_deg": MAX_STRAIGHT_PATH_ERROR_DEG,
            "minimum_path_audit_m": MIN_STRAIGHT_PATH_AUDIT_M,
        },
        "correction_turn_direction": correction_direction,
        "correction_turn_angle_deg": correction_angle_deg,
        "correction_method": correction_method,
        "correction_alignment_axis": "yaw",
        "correction_yaw_deltas_deg": {
            robot_id: alignment_delta_deg if robot_id == alignment_robot_id else 0.0
            for robot_id in ("front", "back")
        },
        "contract": (
            "forward/backward follow the persistent longitudinal axis; left/right "
            "follow its perpendicular lateral axis; all preserve body heading"
        ),
    }


def _translation_semantic_direction(trace: ExecutionTrace, action: str) -> str:
    direct = {
        "carry_forward": "forward",
        "carry_backward": "backward",
        "carry_lateral_left": "left",
        "carry_lateral_right": "right",
    }.get(action, "")
    if direct:
        return direct
    if action == COORDINATED_LINEAR_TRANSPORT:
        return _requested_parameter(trace, action, "direction").strip().lower()
    return ""


def _robot_has_pose_and_yaw(robot: dict[str, Any]) -> bool:
    root_pose = robot.get("root_pose", {})
    pose = root_pose if isinstance(root_pose, dict) and root_pose else robot
    return all(key in pose for key in ("x", "y", "yaw_rad")) or (
        all(key in pose for key in ("x", "y")) and "yaw_deg" in pose
    )


def _synchronization_metrics(
    task: BoxCarryTask,
    before: dict[str, Any] | None,
    after: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Measure per-step dual-robot formation drift and payload yaw drift."""

    empty_metrics = {
        "front_step_displacement_m": 0.0,
        "back_step_displacement_m": 0.0,
        "front_back_forward_displacement_desync_m": 0.0,
        "front_back_step_vector_desync_m": 0.0,
        "payload_step_yaw_drift_deg": 0.0,
        "payload_step_signed_yaw_change_deg": 0.0,
        "front_step_yaw_change_deg": 0.0,
        "back_step_yaw_change_deg": 0.0,
        "front_step_signed_yaw_change_deg": 0.0,
        "back_step_signed_yaw_change_deg": 0.0,
        "team_step_signed_yaw_change_deg": 0.0,
        "max_robot_step_yaw_change_deg": 0.0,
        "robot_heading_disagreement_before_deg": 0.0,
        "robot_heading_disagreement_after_deg": 0.0,
        "robot_heading_disagreement_improvement_deg": 0.0,
        "robot_pair_longitudinal_stagger_m": 0.0,
        "robot_pair_longitudinal_stagger_before_m": 0.0,
        "robot_pair_longitudinal_stagger_improvement_m": 0.0,
        "robot_pair_xy_separation_before_m": 0.0,
        "robot_pair_xy_separation_improvement_m": 0.0,
        "robot_pair_lateral_span_m": 0.0,
        "robot_pair_formation_skew_before_deg": 0.0,
        "robot_pair_formation_skew_deg": 0.0,
        "robot_pair_formation_skew_improvement_deg": 0.0,
        "payload_formation_slip_m": 0.0,
        "payload_step_displacement_xy_m": 0.0,
        "mean_robot_step_displacement_m": 0.0,
        "robot_payload_translation_tracking_ratio": 1.0,
        "robot_pair_separation_along_active_goal_m": 0.0,
        "robot_pair_separation_cross_active_goal_m": 0.0,
        "robot_pair_axis_alignment_active_goal": 0.0,
        "payload_long_axis_alignment_active_goal": 0.0,
        "payload_long_axis_alignment_before_active_goal": 0.0,
        "payload_long_axis_alignment_active_goal_improvement": 0.0,
        "team_forward_alignment_active_goal": 0.0,
        "team_forward_alignment_before_active_goal": 0.0,
        "team_forward_alignment_active_goal_improvement": 0.0,
    }
    if not before:
        return empty_metrics, {"available": False, "reason": "before_state_missing"}

    before_robots = before.get("robots", {})
    after_robots = after.get("robots", {})
    if not isinstance(before_robots, dict) or not isinstance(after_robots, dict):
        return empty_metrics, {"available": False, "reason": "robot_state_missing"}

    displacements: dict[str, tuple[float, float]] = {}
    for robot_id in ("front", "back"):
        before_robot = before_robots.get(robot_id, {})
        after_robot = after_robots.get(robot_id, {})
        if not isinstance(before_robot, dict) or not isinstance(after_robot, dict):
            return empty_metrics, {
                "available": False,
                "reason": f"{robot_id}_state_missing",
            }
        before_x, before_y = _robot_xy(before_robot)
        after_x, after_y = _robot_xy(after_robot)
        displacements[robot_id] = (after_x - before_x, after_y - before_y)

    before_payload = before.get("payload", {})
    after_payload = after.get("payload", {})
    if not isinstance(before_payload, dict) or not isinstance(after_payload, dict):
        return empty_metrics, {"available": False, "reason": "payload_state_missing"}

    goal_x, goal_y = _active_goal_xy(task, before_payload)
    goal_dx = goal_x - _float(before_payload.get("x"), 0.0)
    goal_dy = goal_y - _float(before_payload.get("y"), 0.0)
    goal_norm = hypot(goal_dx, goal_dy)
    if goal_norm > 1.0e-9:
        forward = (goal_dx / goal_norm, goal_dy / goal_norm)
    else:
        forward = (1.0, 0.0)

    front_dx, front_dy = displacements["front"]
    back_dx, back_dy = displacements["back"]
    front_after = after_robots["front"]
    back_after = after_robots["back"]
    front_before = before_robots["front"]
    back_before = before_robots["back"]
    robot_signed_yaw_changes = {
        robot_id: degrees(
            _wrapped_angle(
                _robot_yaw_rad(after_robots[robot_id])
                - _robot_yaw_rad(before_robots[robot_id])
            )
        )
        for robot_id in ("front", "back")
    }
    robot_yaw_changes = {
        robot_id: abs(value)
        for robot_id, value in robot_signed_yaw_changes.items()
    }
    robot_heading_disagreement_before = abs(
        degrees(
            _wrapped_angle(
                _robot_yaw_rad(before_robots["back"])
                - _robot_yaw_rad(before_robots["front"])
            )
        )
    )
    robot_heading_disagreement_after = abs(
        degrees(
            _wrapped_angle(
                _robot_yaw_rad(after_robots["back"])
                - _robot_yaw_rad(after_robots["front"])
            )
        )
    )
    front_before_x, front_before_y = _robot_xy(front_before)
    back_before_x, back_before_y = _robot_xy(back_before)
    front_after_x, front_after_y = _robot_xy(front_after)
    back_after_x, back_after_y = _robot_xy(back_after)
    pair_before_dx = back_before_x - front_before_x
    pair_before_dy = back_before_y - front_before_y
    pair_dx = back_after_x - front_after_x
    pair_dy = back_after_y - front_after_y
    mean_heading_before = atan2(
        sin(_robot_yaw_rad(front_before)) + sin(_robot_yaw_rad(back_before)),
        cos(_robot_yaw_rad(front_before)) + cos(_robot_yaw_rad(back_before)),
    )
    mean_heading_after = atan2(
        sin(_robot_yaw_rad(front_after)) + sin(_robot_yaw_rad(back_after)),
        cos(_robot_yaw_rad(front_after)) + cos(_robot_yaw_rad(back_after)),
    )
    pair_before_longitudinal = (
        pair_before_dx * cos(mean_heading_before)
        + pair_before_dy * sin(mean_heading_before)
    )
    pair_before_lateral = (
        pair_before_dx * -sin(mean_heading_before)
        + pair_before_dy * cos(mean_heading_before)
    )
    pair_longitudinal = (
        pair_dx * cos(mean_heading_after) + pair_dy * sin(mean_heading_after)
    )
    pair_lateral = (
        pair_dx * -sin(mean_heading_after) + pair_dy * cos(mean_heading_after)
    )
    pair_skew_before_deg = degrees(
        atan2(abs(pair_before_longitudinal), abs(pair_before_lateral))
    )
    pair_skew_deg = degrees(atan2(abs(pair_longitudinal), abs(pair_lateral)))
    pair_before_norm = hypot(pair_before_dx, pair_before_dy)
    pair_before_along_goal = pair_before_dx * forward[0] + pair_before_dy * forward[1]
    pair_before_cross_goal = pair_before_dx * (-forward[1]) + pair_before_dy * forward[0]
    pair_along_goal = pair_dx * forward[0] + pair_dy * forward[1]
    pair_cross_goal = pair_dx * (-forward[1]) + pair_dy * forward[0]
    pair_norm = hypot(pair_dx, pair_dy)
    pair_axis_alignment = abs(pair_along_goal) / pair_norm if pair_norm > 1.0e-9 else 0.0
    payload_dx = _float(after_payload.get("x"), 0.0) - _float(before_payload.get("x"), 0.0)
    payload_dy = _float(after_payload.get("y"), 0.0) - _float(before_payload.get("y"), 0.0)
    mean_robot_dx = 0.5 * (front_dx + back_dx)
    mean_robot_dy = 0.5 * (front_dy + back_dy)
    payload_displacement_xy = hypot(payload_dx, payload_dy)
    mean_robot_displacement = hypot(mean_robot_dx, mean_robot_dy)
    tracking_ratio = (
        mean_robot_displacement / payload_displacement_xy
        if payload_displacement_xy > 1.0e-9
        else 1.0
    )
    front_forward = front_dx * forward[0] + front_dy * forward[1]
    back_forward = back_dx * forward[0] + back_dy * forward[1]
    before_yaw = _payload_yaw_rad(before_payload)
    after_yaw = _payload_yaw_rad(after_payload)
    payload_axis_alignment = abs(
        cos(after_yaw) * forward[0] + sin(after_yaw) * forward[1]
    )
    payload_axis_alignment_before = abs(
        cos(before_yaw) * forward[0] + sin(before_yaw) * forward[1]
    )
    team_forward_alignment_before = _team_forward_alignment(
        before_robots,
        forward,
    )
    team_forward_alignment = _team_forward_alignment(
        after_robots,
        forward,
    )
    payload_signed_yaw_change_deg = degrees(
        _wrapped_angle(after_yaw - before_yaw)
    )
    yaw_drift_deg = abs(payload_signed_yaw_change_deg)
    team_signed_yaw_change_deg = degrees(
        _wrapped_angle(mean_heading_after - mean_heading_before)
    )

    metrics = {
        "front_step_displacement_m": hypot(front_dx, front_dy),
        "back_step_displacement_m": hypot(back_dx, back_dy),
        "front_back_forward_displacement_desync_m": abs(front_forward - back_forward),
        "front_back_step_vector_desync_m": hypot(front_dx - back_dx, front_dy - back_dy),
        "payload_step_yaw_drift_deg": yaw_drift_deg,
        "payload_step_signed_yaw_change_deg": payload_signed_yaw_change_deg,
        "front_step_yaw_change_deg": robot_yaw_changes["front"],
        "back_step_yaw_change_deg": robot_yaw_changes["back"],
        "front_step_signed_yaw_change_deg": robot_signed_yaw_changes["front"],
        "back_step_signed_yaw_change_deg": robot_signed_yaw_changes["back"],
        "team_step_signed_yaw_change_deg": team_signed_yaw_change_deg,
        "max_robot_step_yaw_change_deg": max(robot_yaw_changes.values()),
        "robot_heading_disagreement_before_deg": robot_heading_disagreement_before,
        "robot_heading_disagreement_after_deg": robot_heading_disagreement_after,
        "robot_heading_disagreement_improvement_deg": (
            robot_heading_disagreement_before - robot_heading_disagreement_after
        ),
        "robot_pair_longitudinal_stagger_m": pair_longitudinal,
        "robot_pair_longitudinal_stagger_before_m": pair_before_longitudinal,
        "robot_pair_longitudinal_stagger_improvement_m": (
            abs(pair_before_longitudinal) - abs(pair_longitudinal)
        ),
        "robot_pair_xy_separation_before_m": pair_before_norm,
        "robot_pair_xy_separation_improvement_m": pair_norm - pair_before_norm,
        "robot_pair_lateral_span_m": pair_lateral,
        "robot_pair_formation_skew_before_deg": pair_skew_before_deg,
        "robot_pair_formation_skew_deg": pair_skew_deg,
        "robot_pair_formation_skew_improvement_deg": (
            pair_skew_before_deg - pair_skew_deg
        ),
        "payload_formation_slip_m": hypot(
            payload_dx - mean_robot_dx,
            payload_dy - mean_robot_dy,
        ),
        "payload_step_displacement_xy_m": payload_displacement_xy,
        "mean_robot_step_displacement_m": mean_robot_displacement,
        "robot_payload_translation_tracking_ratio": tracking_ratio,
        "robot_pair_separation_along_active_goal_m": pair_along_goal,
        "robot_pair_separation_cross_active_goal_m": pair_cross_goal,
        "robot_pair_axis_alignment_active_goal": pair_axis_alignment,
        "payload_long_axis_alignment_active_goal": payload_axis_alignment,
        "payload_long_axis_alignment_before_active_goal": (
            payload_axis_alignment_before
        ),
        "payload_long_axis_alignment_active_goal_improvement": (
            payload_axis_alignment - payload_axis_alignment_before
        ),
        "team_forward_alignment_active_goal": team_forward_alignment,
        "team_forward_alignment_before_active_goal": team_forward_alignment_before,
        "team_forward_alignment_active_goal_improvement": (
            team_forward_alignment - team_forward_alignment_before
        ),
    }
    evidence = {
        "available": True,
        "front_displacement_xy_m": [front_dx, front_dy],
        "back_displacement_xy_m": [back_dx, back_dy],
        "mean_robot_displacement_xy_m": [mean_robot_dx, mean_robot_dy],
        "payload_displacement_xy_m": [payload_dx, payload_dy],
        "active_goal_forward_xy": [forward[0], forward[1]],
        "robot_pair_back_minus_front_xy_m": [pair_dx, pair_dy],
        "robot_pair_xy_separation_before_m": pair_before_norm,
        "robot_pair_xy_separation_m": pair_norm,
        "robot_pair_xy_separation_improvement_m": pair_norm - pair_before_norm,
        "robot_pair_body_longitudinal_stagger_m": pair_longitudinal,
        "robot_pair_body_lateral_span_m": pair_lateral,
        "robot_pair_formation_skew_before_deg": pair_skew_before_deg,
        "robot_pair_formation_skew_deg": pair_skew_deg,
        "robot_pair_formation_skew_improvement_deg": (
            pair_skew_before_deg - pair_skew_deg
        ),
        "robot_pair_separation_before_along_active_goal_m": pair_before_along_goal,
        "robot_pair_separation_before_cross_active_goal_m": pair_before_cross_goal,
        "robot_pair_separation_along_active_goal_m": pair_along_goal,
        "robot_pair_separation_cross_active_goal_m": pair_cross_goal,
        "robot_pair_axis_alignment_active_goal": pair_axis_alignment,
        "payload_long_axis_alignment_active_goal": payload_axis_alignment,
        "payload_long_axis_alignment_before_active_goal": (
            payload_axis_alignment_before
        ),
        "payload_long_axis_alignment_active_goal_improvement": (
            payload_axis_alignment - payload_axis_alignment_before
        ),
        "team_forward_alignment_active_goal": team_forward_alignment,
        "team_forward_alignment_before_active_goal": team_forward_alignment_before,
        "team_forward_alignment_active_goal_improvement": (
            team_forward_alignment - team_forward_alignment_before
        ),
        "front_forward_displacement_m": front_forward,
        "back_forward_displacement_m": back_forward,
        "payload_yaw_before_deg": degrees(before_yaw),
        "payload_yaw_after_deg": degrees(after_yaw),
        "robot_yaw_change_deg": robot_yaw_changes,
        "robot_heading_disagreement_before_deg": robot_heading_disagreement_before,
        "robot_heading_disagreement_after_deg": robot_heading_disagreement_after,
        "robot_heading_disagreement_improvement_deg": (
            robot_heading_disagreement_before - robot_heading_disagreement_after
        ),
    }
    return metrics, evidence


def _endpoint_longitudinal_alignment_evidence(
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Describe one forward-only lagger catch-up without rejecting the checkpoint."""

    signed_stagger = float(metrics["robot_pair_longitudinal_stagger_m"])
    magnitude = abs(signed_stagger)
    required = magnitude > LONGITUDINAL_ALIGNMENT_TRIGGER_M + 1.0e-9
    if signed_stagger > 0.0:
        lagging_robot, reference_robot = "front", "back"
    else:
        lagging_robot, reference_robot = "back", "front"
    correction_delta = (
        min(
            MAX_LONGITUDINAL_ALIGNMENT_DELTA_M,
            max(MIN_LONGITUDINAL_ALIGNMENT_DELTA_M, magnitude),
        )
        if required
        else 0.0
    )
    robot_deltas = {"front": 0.0, "back": 0.0}
    if required:
        robot_deltas[lagging_robot] = correction_delta
    return {
        "available": True,
        "required": required,
        "signed_back_minus_front_stagger_m": signed_stagger,
        "absolute_stagger_m": magnitude,
        "trigger_m": LONGITUDINAL_ALIGNMENT_TRIGGER_M,
        "axis": "longitudinal",
        "lagging_robot": lagging_robot if required else "",
        "reference_robot": reference_robot if required else "",
        "robot_deltas_m": robot_deltas,
        "minimum_delta_m": MIN_LONGITUDINAL_ALIGNMENT_DELTA_M,
        "maximum_delta_m": MAX_LONGITUDINAL_ALIGNMENT_DELTA_M,
        "checkpoint_disposition": (
            "accepted_with_required_followup" if required else "accepted"
        ),
        "contract": (
            "positive stagger means front lags; negative stagger means back lags. "
            "The reference robot holds and only the lagging robot moves body-forward; "
            "negative longitudinal deltas are forbidden."
        ),
    }


def _payload_yaw_rad(payload: dict[str, Any]) -> float:
    if "yaw_rad" in payload:
        return _float(payload.get("yaw_rad"), 0.0)
    return _float(payload.get("yaw_deg"), 0.0) * 3.141592653589793 / 180.0


def _robot_yaw_rad(robot: dict[str, Any]) -> float:
    root_pose = robot.get("root_pose", {})
    if isinstance(root_pose, dict) and "yaw_rad" in root_pose:
        return _float(root_pose.get("yaw_rad"), 0.0)
    if "yaw_rad" in robot:
        return _float(robot.get("yaw_rad"), 0.0)
    return _float(robot.get("yaw_deg"), 0.0) * 3.141592653589793 / 180.0


def _team_forward_alignment(
    robots: dict[str, Any],
    forward: tuple[float, float],
) -> float:
    heading_x = 0.0
    heading_y = 0.0
    count = 0
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            continue
        yaw = _robot_yaw_rad(robot)
        heading_x += cos(yaw)
        heading_y += sin(yaw)
        count += 1
    norm = hypot(heading_x, heading_y)
    if count == 0 or norm <= 1.0e-9:
        return 0.0
    return (heading_x * forward[0] + heading_y * forward[1]) / norm


def _robot_xy(robot: dict[str, Any]) -> tuple[float, float]:
    root_pose = robot.get("root_pose", {})
    root_pose = root_pose if isinstance(root_pose, dict) else {}
    return (
        _float(robot.get("x"), _float(root_pose.get("x"), 0.0)),
        _float(robot.get("y"), _float(root_pose.get("y"), 0.0)),
    )


def _wrapped_angle(angle_rad: float) -> float:
    return (float(angle_rad) + 3.141592653589793) % (2.0 * 3.141592653589793) - 3.141592653589793


def _float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _min_robot_height(robots: dict[str, Any]) -> float:
    heights = []
    for payload in robots.values():
        if isinstance(payload, dict):
            heights.append(_float(payload.get("pelvis_height_m"), 0.0))
    return min(heights) if heights else 0.0


def _robot_pair_xy_separation(robots: dict[str, Any]) -> float | None:
    """Return the horizontal pelvis-root spacing for the loaded robot pair."""

    if not isinstance(robots, dict):
        return None
    front = robots.get("front")
    back = robots.get("back")
    if not isinstance(front, dict) or not isinstance(back, dict):
        return None
    try:
        front_x = float(front["x"])
        front_y = float(front["y"])
        back_x = float(back["x"])
        back_y = float(back["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return hypot(front_x - back_x, front_y - back_y)


def _post_release_lifecycle_state(state: object) -> dict[str, bool]:
    if not isinstance(state, dict):
        lifecycle: dict[str, Any] = {}
    else:
        value = state.get("post_release_lifecycle", {})
        lifecycle = value if isinstance(value, dict) else {}
    return {
        "payload_released": bool(lifecycle.get("payload_released", False)),
        "crouch_latched": bool(lifecycle.get("crouch_latched", False)),
        "crouched_retreat_completed": bool(
            lifecycle.get("crouched_retreat_completed", False)
        ),
        "stand_recovered": bool(lifecycle.get("stand_recovered", False)),
    }


def _robots_in_post_release_crouch(state: object) -> bool:
    if not isinstance(state, dict):
        return False
    robots = state.get("robots", {})
    if not isinstance(robots, dict):
        return False
    crouch_fractions: list[float] = []
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            return False
        crouch = robot.get("place_crouch", {})
        if not isinstance(crouch, dict) or not bool(
            crouch.get("post_release_latched", False)
        ):
            return False
        crouch_fractions.append(_float(crouch.get("crouch"), 0.0))
    return bool(
        len(crouch_fractions) == 2
        and min(crouch_fractions) >= POST_RELEASE_MIN_CROUCH_FRACTION
    )


def _robots_upright_after_release(state: object) -> bool:
    if not isinstance(state, dict):
        return False
    robots = state.get("robots", {})
    if not isinstance(robots, dict):
        return False
    heights = [
        _float(robots.get(robot_id, {}).get("pelvis_height_m"), 0.0)
        for robot_id in ("front", "back")
        if isinstance(robots.get(robot_id), dict)
    ]
    return bool(
        len(heights) == 2
        and min(heights) >= POST_RELEASE_UPRIGHT_MIN_PELVIS_HEIGHT_M
        and not _robots_in_post_release_crouch(state)
    )


def _estimate_forward_progress(state: dict[str, Any]) -> float:
    robots = state.get("robots", {})
    if not isinstance(robots, dict):
        return 0.0
    distances = []
    for payload in robots.values():
        if not isinstance(payload, dict):
            continue
        initial_x = _float(payload.get("initial_x"), _float(payload.get("x"), 0.0))
        initial_y = _float(payload.get("initial_y"), _float(payload.get("y"), 0.0))
        dx = _float(payload.get("x"), initial_x) - initial_x
        dy = _float(payload.get("y"), initial_y) - initial_y
        distances.append(hypot(dx, dy))
    return max(distances) if distances else 0.0


def _target_metrics(task: BoxCarryTask, payload: dict[str, Any], robots: dict[str, Any]) -> dict[str, float]:
    payload_x = _float(payload.get("x"), 0.0)
    payload_y = _float(payload.get("y"), 0.0)
    payload_z = _float(payload.get("z"), 0.0)
    target_top_z = task.target_center_z_m + abs(task.target_scale_z_m) / 2.0
    payload_target_z = target_top_z + task.payload_half_height_m
    xy_error = hypot(payload_x - task.target_center_x_m, payload_y - task.target_center_y_m)
    z_error = abs(payload_z - payload_target_z)
    payload_tilt_available = "tilt_deg" in payload
    payload_tilt_deg = _float(payload.get("tilt_deg"), 180.0)
    xy_within_tolerance = xy_error <= task.target_xy_tolerance_m
    target_support_radius_m = 0.5 * min(
        abs(float(task.target_scale_x_m)),
        abs(float(task.target_scale_y_m)),
    )
    com_over_target_support = xy_error <= target_support_radius_m + 1.0e-9
    z_within_tolerance = z_error <= task.target_z_tolerance_m
    terminal_tilt_within_tolerance = bool(
        (
            payload_tilt_available
            and payload_tilt_deg <= task.max_terminal_payload_tilt_deg
        )
        or (
            not payload_tilt_available
            and not task.require_terminal_payload_tilt_measurement
        )
    )
    min_robot_distance = _min_robot_target_distance(task, robots)
    contacts = payload.get("contacts", {}) if isinstance(payload.get("contacts"), dict) else {}
    robot_contacts = set(contacts.get("robots", ()))
    support_robot_contacts = set(contacts.get("support_robots", ()))
    target_contact = bool(contacts.get("target", False))
    payload_released = bool(payload.get("released", False))
    payload_released_on_target = (
        payload_released
        and target_contact
        and not bool(contacts.get("floor", False))
        and not bool(contacts.get("walls", ()))
    )
    payload_on_target = (
        payload_released_on_target
        and not robot_contacts
        and not support_robot_contacts
        # Exact simulator contact proves which object carries the payload.  The
        # COM must still project inside the physical support footprint, while
        # the tighter target_xy_tolerance remains an alignment-quality metric.
        and com_over_target_support
        and z_within_tolerance
        and terminal_tilt_within_tolerance
    )
    robots_clear = (
        min_robot_distance >= task.robot_exit_min_distance_from_target_m
        and not robot_contacts
    )
    pre_stand_min_distance = (
        task.robot_exit_min_distance_from_target_m
        + task.robot_pre_stand_clearance_margin_m
    )
    robots_clear_for_stand = (
        min_robot_distance + 1.0e-9 >= pre_stand_min_distance
        and not robot_contacts
    )
    return {
        "target_center_x_m": float(task.target_center_x_m),
        "target_center_y_m": float(task.target_center_y_m),
        "target_center_z_m": float(task.target_center_z_m),
        "target_top_z_m": float(target_top_z),
        "payload_target_z_m": float(payload_target_z),
        "payload_target_xy_error_m": float(xy_error),
        "payload_target_z_error_m": float(z_error),
        "payload_target_xy_within_tolerance": float(xy_within_tolerance),
        "payload_target_support_radius_m": float(target_support_radius_m),
        "payload_target_com_over_support": float(com_over_target_support),
        "payload_target_z_within_tolerance": float(z_within_tolerance),
        "payload_terminal_tilt_deg": float(payload_tilt_deg),
        "payload_terminal_tilt_available": float(payload_tilt_available),
        "payload_terminal_tilt_within_tolerance": float(
            terminal_tilt_within_tolerance
        ),
        "payload_target_contact": float(target_contact),
        "payload_released_on_target": float(payload_released_on_target),
        "payload_on_target": float(payload_on_target),
        "min_robot_target_distance_m": float(min_robot_distance),
        "robots_clear_after_place": float(robots_clear),
        "robot_pre_stand_min_distance_m": float(pre_stand_min_distance),
        "robots_clear_for_stand": float(robots_clear_for_stand),
    }


def _active_goal_xy(
    task: BoxCarryTask,
    payload: dict[str, Any] | None = None,
) -> tuple[float, float]:
    if task.route_discovery and task.task_stage in {"stage3_right", "full_scene"}:
        payload = payload or {}
        payload_x = _float(payload.get("x"), 0.0)
        payload_y = _float(payload.get("y"), 0.0)
        prefix = _route_topology_prefix(task, [(payload_x, payload_y)])
        if prefix < len(task.route_progress_gates):
            gate = task.route_progress_gates[prefix]
            normal_x, normal_y = _unit_gate_normal(gate)
            remaining = float(gate.threshold_m) - (
                normal_x * payload_x + normal_y * payload_y
            )
            return (
                payload_x + remaining * normal_x,
                payload_y + remaining * normal_y,
            )
        return float(task.target_center_x_m), float(task.target_center_y_m)
    if task.task_stage == "stage1_straight":
        return float(task.stage1_goal_x_m), float(task.stage1_goal_y_m)
    if task.task_stage == "stage2_corridor":
        return (
            float(task.stage2_corridor_goal_x_m),
            float(task.stage2_corridor_goal_y_m),
        )
    if task.task_stage in {"stage3_right", "full_scene"}:
        phase = _scene_route_phase(task, payload or {})
        if phase == "stage1_first_straight":
            return float(task.stage1_goal_x_m), float(task.stage1_goal_y_m)
        if phase == "stage2_long_corridor":
            return (
                float(task.stage2_corridor_goal_x_m),
                float(task.stage2_corridor_goal_y_m),
            )
        return float(task.stage3_right_goal_x_m), float(task.stage3_right_goal_y_m)
    return float(task.target_center_x_m), float(task.target_center_y_m)


def _active_goal_metrics(task: BoxCarryTask, payload: dict[str, Any]) -> dict[str, float]:
    goal_x, goal_y = _active_goal_xy(task, payload)
    delta_x = goal_x - _float(payload.get("x"), 0.0)
    delta_y = goal_y - _float(payload.get("y"), 0.0)
    distance = hypot(delta_x, delta_y)
    tolerance = _active_goal_tolerance(task, payload)
    return {
        "active_stage_goal_x_m": goal_x,
        "active_stage_goal_y_m": goal_y,
        "active_stage_goal_delta_x_m": float(delta_x),
        "active_stage_goal_delta_y_m": float(delta_y),
        "active_stage_goal_distance_m": float(distance),
        "active_stage_goal_tolerance_m": tolerance,
        "active_stage_goal_reached": float(distance <= tolerance),
    }


def _min_robot_target_distance(task: BoxCarryTask, robots: dict[str, Any]) -> float:
    distances = []
    for payload in robots.values():
        if not isinstance(payload, dict):
            continue
        x = _float(payload.get("x"), 0.0)
        y = _float(payload.get("y"), 0.0)
        distances.append(hypot(x - task.target_center_x_m, y - task.target_center_y_m))
    return min(distances) if distances else 0.0


def _target_evidence(task: BoxCarryTask) -> dict[str, Any]:
    return {
        "require_payload_on_target": task.require_payload_on_target,
        "require_physical_target_contact": True,
        "require_payload_released": True,
        "require_robots_exit": task.require_robots_exit,
        "target_translate": [task.target_center_x_m, task.target_center_y_m, task.target_center_z_m],
        "target_scale": [task.target_scale_x_m, task.target_scale_y_m, task.target_scale_z_m],
        "target_top_z_m": task.target_center_z_m + abs(task.target_scale_z_m) / 2.0,
        "payload_center_target_z_m": task.target_center_z_m + abs(task.target_scale_z_m) / 2.0 + task.payload_half_height_m,
        "target_xy_tolerance_m": task.target_xy_tolerance_m,
        "target_z_tolerance_m": task.target_z_tolerance_m,
        "max_terminal_payload_tilt_deg": task.max_terminal_payload_tilt_deg,
        "require_terminal_payload_tilt_measurement": task.require_terminal_payload_tilt_measurement,
        "robot_exit_min_distance_from_target_m": task.robot_exit_min_distance_from_target_m,
    }


def _active_goal_evidence(task: BoxCarryTask, payload: dict[str, Any]) -> dict[str, Any]:
    goal_x, goal_y = _active_goal_xy(task, payload)
    payload_x = _float(payload.get("x"), 0.0)
    payload_y = _float(payload.get("y"), 0.0)
    return {
        "task_stage": task.task_stage,
        "route_phase": _scene_route_phase(task, payload),
        "goal_xy": [goal_x, goal_y],
        "payload_xy": [payload_x, payload_y],
        "goal_delta_xy": [goal_x - payload_x, goal_y - payload_y],
        "tolerance_m": _active_goal_tolerance(task, payload),
    }


def _active_goal_tolerance(task: BoxCarryTask, payload: dict[str, Any]) -> float:
    if task.route_discovery and task.task_stage in {"stage3_right", "full_scene"}:
        return float(task.min_forward_progress_m)
    if task.task_stage == "stage1_straight":
        return float(task.stage1_goal_tolerance_m)
    if task.task_stage == "stage2_corridor":
        return float(task.stage2_corridor_goal_tolerance_m)
    if task.task_stage in {"stage3_right", "full_scene"}:
        phase = _scene_route_phase(task, payload)
        if phase == "stage1_first_straight":
            return float(task.stage1_goal_tolerance_m)
        if phase == "stage2_long_corridor":
            return float(task.stage2_corridor_goal_tolerance_m)
        return float(task.stage3_right_goal_tolerance_m)
    return float(task.target_xy_tolerance_m)


def _scene_route_phase(task: BoxCarryTask, payload: dict[str, Any]) -> str:
    if task.task_stage == "stage2_corridor":
        return "stage2_corridor_transport"
    if task.task_stage not in {"stage3_right", "full_scene"}:
        return task.task_stage
    payload_x = _float(payload.get("x"), 0.0)
    payload_y = _float(payload.get("y"), 0.0)
    if task.route_discovery:
        prefix = _route_topology_prefix(task, [(payload_x, payload_y)])
        if prefix == 0:
            return "route_phase_0"
        if prefix <= len(task.route_progress_gates):
            return task.route_progress_gates[prefix - 1].phase_name
        return "payload_placement"
    if payload_y >= task.stage2_corridor_goal_y_m - task.stage2_corridor_goal_tolerance_m:
        return "stage3_final_straight"
    if payload_x >= task.stage1_goal_x_m - task.stage1_goal_tolerance_m:
        return "stage2_long_corridor"
    return "stage1_first_straight"


def trusted_route_phase_guard(
    task: BoxCarryTask,
    state: dict[str, Any] | None,
    *,
    accepted_steps: object = (),
    include_candidate_state: bool = False,
) -> dict[str, Any]:
    """Cap semantic phase completion using the trusted physical checkpoint.

    The guard exposes only a topological frontier to model roles. Intermediate
    waypoint coordinates remain verifier-owned and are not supplied as route
    answers to the planner.
    """

    if task.task_stage not in {"stage3_right", "full_scene"}:
        return {}
    runtime = state if isinstance(state, dict) else {}
    payload = runtime.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    if not bool(payload.get("active", False)):
        return {
            "source": "trusted_checkpoint_physics",
            "current_topological_phase": "route_phase_0",
            "maximum_completed_prefix": 0,
            "route_geometry_contract_available": True,
            "rule": (
                "payload initialization cannot complete a route phase"
            ),
        }

    lifecycle = _post_release_lifecycle_state(runtime)
    route_gate_count = len(task.route_progress_gates)
    target_region_prefix = route_gate_count + 1
    mission_complete_prefix = route_gate_count + 2
    phase_anchors: list[dict[str, Any]] = []
    current_state_route_eligible = False
    if lifecycle["stand_recovered"]:
        phase = "mission_complete"
        maximum_prefix = mission_complete_prefix
    else:
        contacts = payload.get("contacts", {})
        contacts = contacts if isinstance(contacts, dict) else {}
        target_contact = bool(
            payload.get("target_contact", False)
            or contacts.get("target", False)
        )
        all_steps = (
            list(accepted_steps)
            if isinstance(accepted_steps, (list, tuple))
            else []
        )
        route_progress_steps = _route_progress_accepted_steps(all_steps)
        current_state_route_eligible = bool(
            include_candidate_state
            or not all_steps
            or _accepted_step_is_route_progress(all_steps[-1])
        )
        current_payload_xy = (
            _float(payload.get("x"), 0.0),
            _float(payload.get("y"), 0.0),
        )
        topology = trusted_sequential_route_topology(
            task,
            accepted_steps=route_progress_steps,
            candidate_payload_xy=(
                current_payload_xy if include_candidate_state else None
            ),
            candidate_checkpoint_id=str(
                runtime.get("checkpoint", {}).get("checkpoint_id", "")
                if isinstance(runtime.get("checkpoint"), dict)
                else ""
            ),
        )
        maximum_prefix = int(topology["maximum_completed_prefix"])
        phase_anchors = list(topology["phase_anchors"])
        # Contact may confirm only the target-region gate after every route gate
        # was completed sequentially. It can never skip an unfinished gate.
        if (
            maximum_prefix == route_gate_count
            and current_state_route_eligible
            and target_contact
        ):
            maximum_prefix = target_region_prefix
        if maximum_prefix == 0:
            phase = "route_phase_0"
        elif maximum_prefix <= route_gate_count:
            phase = task.route_progress_gates[maximum_prefix - 1].phase_name
        elif maximum_prefix == target_region_prefix:
            phase = "payload_placement"
        else:
            phase = "mission_complete"

    return {
        "source": "trusted_checkpoint_physics",
        "current_topological_phase": phase,
        "maximum_completed_prefix": maximum_prefix,
        "phase_anchors": phase_anchors,
        "current_state_route_eligible": current_state_route_eligible,
        "route_geometry_contract_available": True,
        "route_gate_count": route_gate_count,
        "rule": (
            "each accepted route-progress checkpoint may advance at most the "
            "active physical gate; setup checkpoints and model-written phase "
            "labels are not completion evidence"
        ),
    }


def trusted_sequential_route_topology(
    task: BoxCarryTask,
    *,
    accepted_steps: object,
    candidate_payload_xy: tuple[float, float] | None = None,
    candidate_checkpoint_id: str = "",
) -> dict[str, Any]:
    """Build the authoritative one-gate-per-progress-checkpoint frontier."""

    records: list[dict[str, Any]] = []
    steps = _route_progress_accepted_steps(accepted_steps)
    for step in steps:
        measured = step.get("measured_outcome", step)
        measured = measured if isinstance(measured, dict) else {}
        try:
            records.append(
                {
                    "x": float(measured["payload_x_m"]),
                    "y": float(measured["payload_y_m"]),
                    "checkpoint_after": str(step.get("checkpoint_after", "")),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    if candidate_payload_xy is not None:
        records.append(
            {
                "x": float(candidate_payload_xy[0]),
                "y": float(candidate_payload_xy[1]),
                "checkpoint_after": str(candidate_checkpoint_id),
            }
        )
    anchors: list[dict[str, Any]] = []
    prefix = 0
    previous_route_progress_xy: tuple[float, float] | None = None
    for record in records:
        current_xy = (float(record["x"]), float(record["y"]))
        next_prefix = advance_trusted_route_prefix(
            task,
            prefix=prefix,
            previous_route_progress_xy=previous_route_progress_xy,
            current_route_progress_xy=current_xy,
        )
        if next_prefix > prefix:
            anchors.append(
                {
                    "completed_prefix": next_prefix,
                    "payload_x_m": round(float(record["x"]), 6),
                    "payload_y_m": round(float(record["y"]), 6),
                    "checkpoint_after": str(record["checkpoint_after"]),
                }
            )
        prefix = next_prefix
        previous_route_progress_xy = current_xy
    return {
        "maximum_completed_prefix": prefix,
        "phase_anchors": anchors,
        "latest_route_progress_xy": (
            list(previous_route_progress_xy)
            if previous_route_progress_xy is not None
            else []
        ),
        "route_progress_endpoint_count": len(records),
        "semantics_version": 1,
    }


def advance_trusted_route_prefix(
    task: BoxCarryTask,
    *,
    prefix: int,
    previous_route_progress_xy: tuple[float, float] | None,
    current_route_progress_xy: tuple[float, float],
) -> int:
    """Advance at most the active gate for one accepted progress endpoint."""

    current = max(0, int(prefix))
    x, y = current_route_progress_xy
    if current < len(task.route_progress_gates):
        gate = task.route_progress_gates[current]
        normal_x, normal_y = _unit_gate_normal(gate)
        after_progress = normal_x * x + normal_y * y
        before_progress = (
            None
            if previous_route_progress_xy is None
            else normal_x * previous_route_progress_xy[0]
            + normal_y * previous_route_progress_xy[1]
        )
        crossed = bool(
            after_progress + 1.0e-9 >= float(gate.threshold_m)
            and (
                before_progress is None
                or before_progress + 1.0e-9 < float(gate.threshold_m)
            )
        )
        if crossed:
            return current + 1
        return current
    if current == len(task.route_progress_gates):
        tolerance = max(
            task.target_xy_tolerance_m,
            task.stage3_right_goal_tolerance_m,
        )
        after_distance = hypot(
            x - task.target_center_x_m,
            y - task.target_center_y_m,
        )
        before_distance = (
            None
            if previous_route_progress_xy is None
            else hypot(
                previous_route_progress_xy[0] - task.target_center_x_m,
                previous_route_progress_xy[1] - task.target_center_y_m,
            )
        )
        if after_distance <= tolerance and (
            before_distance is None or before_distance > tolerance
        ):
            return current + 1
    return current


def _route_topology_anchors(
    task: BoxCarryTask,
    *,
    accepted_steps: object,
    current_payload_xy: tuple[float, float] | None,
    current_checkpoint_id: str,
) -> list[dict[str, Any]]:
    """Compatibility wrapper over the authoritative sequential frontier."""

    return list(
        trusted_sequential_route_topology(
            task,
            accepted_steps=accepted_steps,
            candidate_payload_xy=current_payload_xy,
            candidate_checkpoint_id=current_checkpoint_id,
        )["phase_anchors"]
    )


def _route_progress_accepted_steps(accepted_steps: object) -> list[dict[str, Any]]:
    """Exclude restorable setup checkpoints from route chronology consumers."""

    steps = accepted_steps if isinstance(accepted_steps, (list, tuple)) else ()
    return [
        step
        for step in steps
        if isinstance(step, dict) and _accepted_step_is_route_progress(step)
    ]


def _accepted_step_is_route_progress(step: object) -> bool:
    if not isinstance(step, dict):
        return False
    outcome = str(step.get("transition_outcome", "")).strip()
    if outcome:
        return outcome in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
    # Older manifests used this learning flag before transition outcomes were
    # explicit. New topology state is never conditional on memory promotion.
    return step.get("positive_route_memory_eligible") is True


def _accepted_payload_xy_trajectory(accepted_steps: object) -> list[tuple[float, float]]:
    steps = _route_progress_accepted_steps(accepted_steps)
    trajectory: list[tuple[float, float]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        measured = step.get("measured_outcome", step)
        measured = measured if isinstance(measured, dict) else {}
        try:
            trajectory.append(
                (float(measured["payload_x_m"]), float(measured["payload_y_m"]))
            )
        except (KeyError, TypeError, ValueError):
            continue
    return trajectory


def _route_topology_prefix(
    task: BoxCarryTask,
    trajectory: list[tuple[float, float]],
) -> int:
    """Project coordinates for legacy instantaneous scene diagnostics.

    Formal trusted chronology uses :func:`trusted_sequential_route_topology`;
    this coordinate-only helper intentionally remains non-authoritative.
    """

    prefix = 0
    for payload_x, payload_y in trajectory:
        while prefix < len(task.route_progress_gates):
            gate = task.route_progress_gates[prefix]
            normal_x, normal_y = _unit_gate_normal(gate)
            progress = normal_x * payload_x + normal_y * payload_y
            if progress + 1.0e-9 < float(gate.threshold_m):
                break
            prefix += 1
        if prefix >= len(task.route_progress_gates) and hypot(
            payload_x - task.target_center_x_m,
            payload_y - task.target_center_y_m,
        ) <= max(task.target_xy_tolerance_m, task.stage3_right_goal_tolerance_m):
            prefix = len(task.route_progress_gates) + 1
    return prefix


def _unit_gate_normal(gate: Any) -> tuple[float, float]:
    normal_x = float(gate.normal_x)
    normal_y = float(gate.normal_y)
    magnitude = hypot(normal_x, normal_y)
    if magnitude <= 1.0e-9:
        raise ValueError(f"route progress gate {gate.phase_name!r} has zero normal")
    return normal_x / magnitude, normal_y / magnitude


def _visual_summary(report: VisualVerifierReport | None) -> str:
    if report is None:
        return ""
    return report.summary


def _visual_evidence(report: VisualVerifierReport | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        "visual_verifier": {
            "status": report.status,
            "frame_path": report.frame_path,
            "model_name": report.model_name,
            "failure_type": report.failure_type,
            "confidence": report.confidence,
            "summary": report.summary,
            "evidence": report.evidence,
            "recommendations": list(report.recommendations),
        }
    }


def _route_regression_evidence(
    report: VisualVerifierReport | None,
    *,
    action: str,
    metrics: dict[str, float],
    use_target_progress: bool = True,
    local_navigation_transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require measured motion plus physical or visible-clearance corroboration."""

    if (
        report is None
        or report.status != "ok"
        or action
        not in (ROUTE_TRANSLATION_ACTIONS | CONTINUOUS_TURN_ACTIONS)
    ):
        return {}
    visual = report.evidence if isinstance(report.evidence, dict) else {}
    local_transition = (
        local_navigation_transition
        if isinstance(local_navigation_transition, dict)
        else {}
    )
    if (
        action in CONTINUOUS_TURN_ACTIONS
        and local_transition.get("available", False)
        and local_transition.get("semantic_heading_improved", False)
    ):
        # A declared local turn is scored by measured heading-error reduction.
        # Camera route labels remain useful diagnostics, but cannot override the
        # exact pose transition unless physics also reports wall contact.
        if _float(metrics.get("wall_contact_count"), 0.0) <= 0.0:
            return {}
    if not bool(visual.get("route_direction_risk_confirmed", False)):
        return {}
    displacement = _float(metrics.get("payload_step_displacement_xy_m"), 0.0)
    if displacement < MIN_MEANINGFUL_TRANSLATION_M:
        return {}
    target_progress = _float(metrics.get("payload_target_progress_m"), 0.0)
    wall_contact_count = _float(metrics.get("wall_contact_count"), 0.0)
    clearance_changes = {
        key: _float(metrics.get(key), 0.0)
        for key in (
            "min_wall_clearance_change_m",
            "payload_wall_clearance_change_m",
            "team_wall_clearance_change_m",
        )
        if key in metrics
    }
    reduced_clearance = any(
        change <= -MIN_MEANINGFUL_TRANSLATION_M
        for change in clearance_changes.values()
    )
    physical_regression = bool(
        use_target_progress
        and target_progress <= -MIN_MEANINGFUL_TRANSLATION_M
    )
    visible_near_wall_regression = bool(
        visual.get("payload_wall_contact_or_high_risk", False)
        and str(visual.get("team_footprint_clearance", "")).strip().lower()
        in {"tight", "blocked"}
        and str(
            visual.get("team_heading_relative_to_visible_free_space", "")
        ).strip().lower()
        == "toward_wall_or_blocked_space"
    )
    if not (
        physical_regression
        or wall_contact_count > 0.0
        or reduced_clearance
        or visible_near_wall_regression
    ):
        # Camera-space direction labels are advisory. A model cannot turn screen-left
        # or screen-right into a world/body-frame route failure when exact MuJoCo
        # state reports safe progress and the global view does not also show a tight
        # near-wall footprint. Euclidean target progress alone cannot validate a
        # motion that is visibly driving a long payload into an intervening wall.
        return {}
    basis = str(visual.get("route_direction_risk_basis", "uncertain"))
    if basis not in {
        "executed_motion_into_blocked_space",
        "unexplained_retreat",
    }:
        return {}
    return {
        "decision": "reject_executed_route_direction",
        "failure_scope": "checkpoint_local_semantic_motion_objective",
        "risk_basis": basis,
        "route_direction_evidence": str(
            visual.get("route_direction_evidence", "")
        ).strip(),
        "team_heading_relative_to_target": visual.get(
            "team_heading_relative_to_target", "uncertain"
        ),
        "team_heading_relative_to_visible_free_space": visual.get(
            "team_heading_relative_to_visible_free_space", "uncertain"
        ),
        "rollout_motion_hint": visual.get("rollout_motion_hint", {}),
        "payload_step_displacement_xy_m": displacement,
        "payload_target_progress_m": target_progress,
        "target_progress_used_for_route_decision": bool(use_target_progress),
        "route_discovery": not bool(use_target_progress),
        "wall_contact_count": wall_contact_count,
        "wall_clearance_changes_m": clearance_changes,
        "physics_corroboration": {
            "target_regression": physical_regression,
            "wall_contact": wall_contact_count > 0.0,
            "reduced_wall_clearance": reduced_clearance,
            "visible_near_wall_regression": visible_near_wall_regression,
        },
        "detour_policy": (
            "target-away motion remains legal when visible obstacle evidence explains a detour"
        ),
    }


def _is_final_route_approach_phase(value: object) -> bool:
    phase_id = str(value).strip().lower()
    return "final_straight" in phase_id
