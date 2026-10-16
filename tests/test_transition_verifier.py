from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest

from praxis_coop.cap_policy.contracts import (
    BoxCarryTask,
    EvaluationReport,
    ExecutionTrace,
    PolicyScript,
    RobotPolicyScript,
    RouteProgressGate,
    RouteSegment,
    SkillCallRecord,
)
from praxis_coop.cap_policy.coordination_specs import (
    COORDINATED_LINEAR_DISTANCE_CHOICES_M,
    MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG,
    coordination_primitive_method_spec,
    validate_coordination_primitive_parameters,
)
from praxis_coop.cap_policy.model_agents import compact_runtime_state
from praxis_coop.cap_policy.model_role_pipeline import _accepted_route_phase_evidence
from praxis_coop.cap_policy.simulation_diagnostics import trusted_route_phase_guard
from praxis_coop.cap_policy.evolution_controller import EvolutionController
from praxis_coop.cap_policy.route_spec import (
    RouteSpecConfigurationError,
    load_task_route_spec,
    route_guidance,
    validate_task_route_spec,
    validate_locked_route_reference_unchanged,
    validate_task_plan_route_reference,
)
from praxis_coop.cap_policy.transition_verifier import (
    TransitionOutcome,
    _phase_anchor_regression,
    apply_transition_decision,
    detect_state_progress_cycle,
    transition_outcome_from_report,
    verify_transition,
)


def _task() -> BoxCarryTask:
    return BoxCarryTask(
        task_id="freeze_gate",
        target="place on target",
        task_stage="full_scene",
        route_discovery=True,
    )


def _route_document(task: BoxCarryTask) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "directed_connected_route",
        "scene_id": task.scene_id,
        "mission_topology": list(task.mission_topology),
        "progress_gates": [asdict(item) for item in task.route_progress_gates],
        "segments": [asdict(item) for item in task.route_segments],
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda doc: doc["segments"].pop(), "segment count"),
        (
            lambda doc: doc["segments"][1].update({"start_x_m": 99.0}),
            "disconnected",
        ),
        (
            lambda doc: doc["segments"][0].update(
                {"center_y_m": doc["segments"][0]["center_y_m"] + 3.0}
            ),
            "centered corridor width",
        ),
        (
            lambda doc: doc["progress_gates"][0].update({"threshold_m": 99.0}),
            "does not reach",
        ),
        (
            lambda doc: doc["progress_gates"][0].update(
                {"normal_x": -1.0, "threshold_m": -1.15}
            ),
            "must face along",
        ),
        (
            lambda doc: doc["segments"][1].update(
                {"center_x_m": 4.0, "half_width_m": 0.1}
            ),
            "does not cover the boundary",
        ),
        (
            lambda doc: doc["segments"][-1].update({"goal_x_m": -0.5}),
            "final segment",
        ),
    ),
)
def test_route_spec_rejects_broken_topology_before_execution(
    tmp_path, mutate, message
):
    task = _task()
    document = _route_document(task)
    mutate(document)
    path = tmp_path / "route.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RouteSpecConfigurationError, match=message):
        load_task_route_spec(task, path)


def test_route_spec_rejects_action_answer_fields(tmp_path):
    task = _task()
    document = _route_document(task)
    document["primitive"] = "coordinated_pivot"
    path = tmp_path / "route.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RouteSpecConfigurationError, match="unsupported fields"):
        load_task_route_spec(task, path)


def test_route_spec_rejects_unknown_document_kind(tmp_path):
    task = _task()
    document = _route_document(task)
    document["kind"] = "action_sequence"
    path = tmp_path / "route.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RouteSpecConfigurationError, match="kind must be"):
        load_task_route_spec(task, path)


def test_route_spec_validation_is_phase_name_neutral():
    task = _task()
    renamed = replace(
        task,
        route_progress_gates=tuple(
            replace(gate, phase_name=f"anonymous_{index}")
            for index, gate in enumerate(task.route_progress_gates)
        ),
    )

    assert validate_task_route_spec(task)["valid"]
    assert validate_task_route_spec(renamed)["valid"]


def test_route_spec_geometry_is_equivariant_to_rotation_and_translation():
    task = _task()
    translation = (8.0, -3.0)

    def transform_point(x: float, y: float) -> tuple[float, float]:
        return -y + translation[0], x + translation[1]

    def rotate_vector(x: float, y: float) -> tuple[float, float]:
        return -y, x

    transformed_gates = []
    for gate in task.route_progress_gates:
        normal_x, normal_y = rotate_vector(gate.normal_x, gate.normal_y)
        transformed_gates.append(
            replace(
                gate,
                normal_x=normal_x,
                normal_y=normal_y,
                threshold_m=(
                    gate.threshold_m
                    + normal_x * translation[0]
                    + normal_y * translation[1]
                ),
            )
        )
    transformed_segments = []
    for segment in task.route_segments:
        direction_x, direction_y = rotate_vector(
            segment.direction_x,
            segment.direction_y,
        )
        start_x, start_y = transform_point(segment.start_x_m, segment.start_y_m)
        goal_x, goal_y = transform_point(segment.goal_x_m, segment.goal_y_m)
        center_x, center_y = transform_point(segment.center_x_m, segment.center_y_m)
        transformed_segments.append(
            replace(
                segment,
                direction_x=direction_x,
                direction_y=direction_y,
                start_x_m=start_x,
                start_y_m=start_y,
                goal_x_m=goal_x,
                goal_y_m=goal_y,
                center_x_m=center_x,
                center_y_m=center_y,
            )
        )
    target_x, target_y = transform_point(
        task.target_center_x_m,
        task.target_center_y_m,
    )
    transformed = replace(
        task,
        target_center_x_m=target_x,
        target_center_y_m=target_y,
        route_progress_gates=tuple(transformed_gates),
        route_segments=tuple(transformed_segments),
    )

    assert validate_task_route_spec(transformed)["valid"]
    payload = {"payload": {"x": -0.75, "y": 2.40}}
    moved_x, moved_y = transform_point(-0.75, 2.40)
    original_guidance = route_guidance(task, payload, completed_prefix=0)
    transformed_guidance = route_guidance(
        transformed,
        {"payload": {"x": moved_x, "y": moved_y}},
        completed_prefix=0,
    )

    assert transformed_guidance["world_direction_xy"] == pytest.approx([0.0, 1.0])
    assert transformed_guidance["measured"]["along_segment_progress_m"] == (
        pytest.approx(original_guidance["measured"]["along_segment_progress_m"])
    )
    assert transformed_guidance["measured"]["cross_track_error_m"] == (
        pytest.approx(original_guidance["measured"]["cross_track_error_m"])
    )


def test_final_route_segment_completes_at_target_not_last_intermediate_gate():
    task = _task()

    intermediate = route_guidance(
        task,
        {"payload": {"x": 1.35, "y": 7.35}},
        completed_prefix=len(task.route_progress_gates) - 1,
    )
    final = route_guidance(
        task,
        {"payload": {"x": 0.85, "y": 7.60}},
        completed_prefix=len(task.route_progress_gates),
    )

    assert intermediate["completion_gate"]["kind"] == "oriented_progress_gate"
    assert final["segment_id"] == task.route_segments[-1].segment_id
    assert final["completion_gate"] == {
        "kind": "terminal_target_region",
        "phase_name": "target_region",
        "center_xy_m": [task.target_center_x_m, task.target_center_y_m],
        "xy_tolerance_m": task.target_xy_tolerance_m,
    }


def _trace(
    action: str,
    parameters: dict[str, object] | None = None,
    *,
    verification_contract: dict[str, object] | None = None,
    signed_progress_m: float | None = None,
) -> ExecutionTrace:
    return ExecutionTrace(
        script_id="test",
        attempt=1,
        status="success",
        skill_calls=(
            SkillCallRecord(
                index=1,
                skill_name=action,
                parameters=parameters or {},
                status="success",
                started_at_unix=0.0,
                finished_at_unix=1.0,
                effective_parameters=(
                    {
                        "coordinated_team_result": {
                            "linear_distance_feedback": {
                                "settled_payload_progress_m": signed_progress_m,
                                "semantic_direction": str(
                                    (parameters or {}).get("direction", "")
                                ),
                            }
                        }
                    }
                    if signed_progress_m is not None
                    else {}
                ),
            ),
        ),
        verification_contract=verification_contract or {},
    )


def _report(
    x: float,
    y: float,
    *,
    evidence: dict[str, object] | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        success=True,
        failure_type="none",
        metrics={
            "payload_x_m": x,
            "payload_y_m": y,
            "payload_step_displacement_xy_m": 0.5,
        },
        evidence=evidence or {},
    )


def _annotated_report(
    outcome: TransitionOutcome,
    *,
    success: bool = True,
    failure_type: str = "none",
) -> EvaluationReport:
    return EvaluationReport(
        success=success,
        failure_type=failure_type,
        evidence={
            "transition_decision": {
                "outcome": outcome.value,
                "trusted_checkpoint_allowed": outcome
                in {
                    TransitionOutcome.ACCEPT_SETUP,
                    TransitionOutcome.ACCEPT_PROGRESS,
                    TransitionOutcome.TASK_COMPLETE,
                },
                "positive_route_memory_allowed": outcome
                in {
                    TransitionOutcome.ACCEPT_PROGRESS,
                    TransitionOutcome.TASK_COMPLETE,
                },
            }
        },
    )


def test_model_skill_schema_uses_bounded_linear_and_pivot_choices():
    assert COORDINATED_LINEAR_DISTANCE_CHOICES_M == (0.25, 0.5, 0.75, 1.0)
    assert MODEL_ROUTE_TURN_ANGLE_CHOICES_DEG == (10.0, 15.0, 20.0, 30.0)
    linear = coordination_primitive_method_spec("coordinated_linear_transport")
    assert linear["parameters"]["distance_m"]["max"] == 1.0
    validate_coordination_primitive_parameters(
        "coordinated_pivot",
        {
            "turn_direction": "counterclockwise",
            "angle_deg": 15.0,
            "travel_direction": "forward",
            "duration_s": 8.0,
            "role": "outer",
            "swing_radius_m": 0.5,
        },
    )


def test_compact_state_serializes_route_neutral_frames_and_immutable_ids():
    compact = compact_runtime_state(
        {
            "payload": {"active": True, "x": 1.0, "y": 2.0, "yaw_deg": 90.0},
            "robots": {
                "front": {"x": 1.0, "y": 1.0, "yaw_deg": 0.0},
                "back": {"x": 1.0, "y": 0.0, "yaw_deg": 180.0},
            },
        }
    )
    assert compact["coordinate_convention"]["yaw"]["positive"].startswith(
        "counterclockwise"
    )
    assert compact["coordinate_convention"]["fixed_global_camera"] == {
        "screen_right": "world -X",
        "screen_down": "world +Y",
        "view_direction": "toward world -Z",
    }
    assert compact["robot_identity_contract"].keys() == {"front", "back"}
    assert compact["payload"]["heading_vector_world_xy"] == [0.0, 1.0]
    assert compact["robots"]["front"]["heading_vector_world_xy"] == [1.0, 0.0]


def test_safe_severe_backward_regression_is_rejected_before_checkpoint():
    manifest = {
        "initial_state": {"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
        "steps": [
            {
                "checkpoint_after": "accepted_step_01",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {"payload_x_m": 1.2, "payload_y_m": 2.4},
            }
        ],
    }
    decision = verify_transition(
        task=_task(),
        report=_report(0.5, 2.4),
        trace=_trace("coordinated_linear_transport"),
        before_state={"payload": {"active": True, "x": 1.2, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 0.5, "y": 2.4}},
        accepted_manifest=manifest,
    )
    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.trusted_checkpoint_allowed is False
    assert decision.evidence["source"] == "phase_anchor_retention"
    rejected = apply_transition_decision(_report(0.5, 2.4), decision)
    assert rejected.success is False
    assert rejected.failure_type == "route_regression"
    assert rejected.evidence["transition_decision"]["outcome"] == (
        "REJECT_REGRESSION"
    )


def _cross_passage_manifest() -> dict[str, object]:
    return {
        "initial_state": {"payload": {"x": -1.2, "y": 2.4, "yaw_deg": 0.0}},
        "steps": [
            {
                "checkpoint_after": "accepted_step_03",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {
                    "payload_x_m": 1.30769,
                    "payload_y_m": 2.397201,
                },
            },
            {
                "checkpoint_after": "accepted_step_10",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {
                    "payload_x_m": 2.164673,
                    "payload_y_m": 3.721254,
                },
            },
        ],
    }


def test_v24_backward_route_transport_cannot_become_setup_checkpoint():
    decision = verify_transition(
        task=_task(),
        report=_report(2.500191, 4.697136),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward"},
        ),
        before_state={
            "payload": {"active": True, "x": 2.875699, "y": 5.574611}
        },
        after_state={
            "payload": {"active": True, "x": 2.500191, "y": 4.697136}
        },
        accepted_manifest=_cross_passage_manifest(),
    )

    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.evidence["source"] == "active_route_progress_regression"
    assert decision.evidence["route_progress_delta_m"] == pytest.approx(-0.877475)


def test_unscoped_route_recovery_cannot_self_authorize_setup():
    decision = verify_transition(
        task=_task(),
        report=_report(2.500191, 4.697136),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward", "coordination_mode": "sync"},
            verification_contract={"route_recovery": {"active": True}},
        ),
        before_state={
            "payload": {"active": True, "x": 2.875699, "y": 5.574611}
        },
        after_state={
            "payload": {"active": True, "x": 2.500191, "y": 4.697136}
        },
        accepted_manifest=_cross_passage_manifest(),
    )

    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.trusted_checkpoint_allowed is False


def test_explicit_recovery_cannot_bypass_completed_phase_anchor():
    decision = verify_transition(
        task=_task(),
        report=_report(0.5, 2.4),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward", "coordination_mode": "sync"},
            verification_contract={
                "route_recovery": {
                    "active": True,
                    "setup_only": True,
                    "semantic_objective": "backward",
                    "phase_completion_authorized": False,
                    "positive_route_memory_eligible": False,
                }
            },
        ),
        before_state={"payload": {"active": True, "x": 1.2, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 0.5, "y": 2.4}},
        accepted_manifest={
            "initial_state": {"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
            "steps": [
                {
                    "checkpoint_after": "accepted_step_01",
                    "transition_outcome": "ACCEPT_PROGRESS",
                    "positive_route_memory_eligible": True,
                    "measured_outcome": {"payload_x_m": 1.2, "payload_y_m": 2.4},
                }
            ],
        },
    )

    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.trusted_checkpoint_allowed is False
    assert decision.evidence["source"] == "phase_anchor_retention"


def test_model_declared_setup_only_does_not_authorize_route_recovery():
    decision = verify_transition(
        task=_task(),
        report=_report(
            1.334836,
            2.718391,
            evidence={},
        ),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward", "distance_m": 0.25, "coordination_mode": "sync"},
            verification_contract={
                "committed_subtask": {
                    "primitive": "coordinated_linear_transport",
                    "semantic_objective": "backward",
                    "coordination_mode": "sync",
                },
                "route_recovery": {
                    "active": True,
                    "setup_only": True,
                    "semantic_objective": "backward",
                    "phase_completion_authorized": False,
                    "positive_route_memory_eligible": False,
                },
            },
        ),
        before_state={"payload": {"active": True, "x": 1.396969, "y": 3.309852}},
        after_state={"payload": {"active": True, "x": 1.334836, "y": 2.718391}},
        accepted_manifest=_cross_passage_manifest(),
    )

    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.trusted_checkpoint_allowed is False


def test_checkpoint_scoped_host_recovery_with_signed_effect_is_setup():
    decision = verify_transition(
        task=_task(),
        report=_report(2.500191, 4.697136),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward", "coordination_mode": "sync"},
            verification_contract={
                "route_recovery": {
                    "active": True,
                    "host_authoritative": True,
                    "source": "host_route_recovery",
                    "scope_checkpoint_id": "accepted_step_10",
                    "semantic_objective": "backward",
                    "setup_only": True,
                    "phase_completion_authorized": False,
                    "positive_route_memory_eligible": False,
                }
            },
            signed_progress_m=0.25,
        ),
        before_state={
            "payload": {"active": True, "x": 2.875699, "y": 5.574611}
        },
        after_state={
            "payload": {"active": True, "x": 2.500191, "y": 4.697136}
        },
        accepted_manifest=_cross_passage_manifest(),
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.trusted_checkpoint_allowed is True
    assert decision.positive_route_memory_allowed is False
    assert decision.evidence["recovery_geometry"]["metric"] == (
        "signed_command_progress"
    )


def test_model_owned_bounded_setup_with_signed_effect_is_accept_setup_only():
    """A validated model setup remains model-owned but cannot become progress."""

    decision = verify_transition(
        task=_task(),
        report=_report(2.500191, 4.697136),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward", "coordination_mode": "sync"},
            verification_contract={
                "committed_subtask": {
                    "source": "host_binding_from_validated_task_orchestrator_plan",
                    "immutable_by_coder": True,
                    "checkpoint_id": "accepted_step_10",
                    "primitive": "coordinated_linear_transport",
                    "semantic_objective": "backward",
                    "coordination_mode": "sync",
                    "navigation_reference": "visible_connected_free_space",
                    "setup_only": True,
                    "phase_completion_authorized": False,
                    "positive_route_memory_eligible": False,
                }
            },
            signed_progress_m=0.25,
        ),
        before_state={
            "payload": {"active": True, "x": 2.875699, "y": 5.574611}
        },
        after_state={
            "payload": {"active": True, "x": 2.500191, "y": 4.697136}
        },
        accepted_manifest=_cross_passage_manifest(),
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.trusted_checkpoint_allowed is True
    assert decision.positive_route_memory_allowed is False
    assert decision.evidence["model_owned_setup_effect"]["improved"] is True
    assert decision.evidence["route_progress"]["progress"] is False


def test_stale_checkpoint_recovery_scope_is_rejected():
    decision = verify_transition(
        task=_task(),
        report=_report(2.500191, 4.697136),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward", "coordination_mode": "sync"},
            verification_contract={
                "route_recovery": {
                    "active": True,
                    "source": "host_route_recovery",
                    "scope_checkpoint_id": "accepted_step_03",
                    "setup_only": True,
                    "phase_completion_authorized": False,
                }
            },
            signed_progress_m=0.25,
        ),
        before_state={
            "payload": {"active": True, "x": 2.875699, "y": 5.574611}
        },
        after_state={
            "payload": {"active": True, "x": 2.500191, "y": 4.697136}
        },
        accepted_manifest=_cross_passage_manifest(),
    )

    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.trusted_checkpoint_allowed is False


def test_checkpoint_scoped_recovery_checks_nested_team_direction():
    contract = {
        "route_recovery": {
            "active": True,
            "host_authoritative": True,
            "source": "host_route_recovery",
            "scope_checkpoint_id": "accepted_step_10",
            "semantic_objective": "backward",
            "setup_only": True,
            "phase_completion_authorized": False,
        }
    }

    def team_trace(feedback_direction: str) -> ExecutionTrace:
        return ExecutionTrace(
            script_id="nested_team_recovery",
            attempt=1,
            status="success",
            skill_calls=(
                SkillCallRecord(
                    index=1,
                    skill_name="coordinated_linear_transport",
                    parameters={
                        "actions": {
                            robot_id: {
                                "method": "coordinated_linear_transport",
                                "parameters": {"direction": "backward"},
                            }
                            for robot_id in ("front", "back")
                        }
                    },
                    effective_parameters={
                        "coordinated_team_result": {
                            "linear_distance_feedback": {
                                "semantic_direction": feedback_direction,
                                "settled_payload_progress_m": 0.25,
                            }
                        }
                    },
                    status="success",
                    started_at_unix=0.0,
                    finished_at_unix=1.0,
                ),
            ),
            verification_contract=contract,
        )

    common = {
        "task": _task(),
        "report": _report(2.500191, 4.697136),
        "before_state": {
            "payload": {"active": True, "x": 2.875699, "y": 5.574611}
        },
        "after_state": {
            "payload": {"active": True, "x": 2.500191, "y": 4.697136}
        },
        "accepted_manifest": _cross_passage_manifest(),
    }
    accepted = verify_transition(trace=team_trace("backward"), **common)
    mismatch = verify_transition(trace=team_trace("forward"), **common)

    assert accepted.outcome is TransitionOutcome.ACCEPT_SETUP
    assert accepted.evidence["recovery_geometry"]["direction_evidence"] == {
        "consistent": True,
        "declared_direction": "backward",
        "executed_directions": ["backward"],
        "feedback_directions": ["backward"],
        "executable_methods": ["coordinated_linear_transport"],
    }
    assert mismatch.outcome is TransitionOutcome.REJECT_REGRESSION
    assert mismatch.evidence["recovery_geometry"]["contract_authoritative"] is False


def test_v53_gpt_unsafe_backward_recovery_still_rejects_formation():
    report = EvaluationReport(
        success=False,
        failure_type="formation_desync",
        metrics={
            "robot_pair_formation_skew_before_deg": 24.485819,
            "robot_pair_formation_skew_deg": 32.892478,
            "wall_contact_count": 0.0,
            "payload_x_m": 1.0,
            "payload_y_m": 3.0,
        },
        evidence={},
    )
    decision = verify_transition(
        task=_task(),
        report=report,
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "backward", "distance_m": 0.5, "coordination_mode": "sync"},
            verification_contract={
                "route_recovery": {
                    "active": True,
                    "setup_only": True,
                    "semantic_objective": "backward",
                }
            },
        ),
        before_state={"payload": {"active": True, "x": 1.4, "y": 3.3}},
        after_state={"payload": {"active": True, "x": 1.1, "y": 2.4}},
        accepted_manifest=_cross_passage_manifest(),
    )

    assert decision.outcome is TransitionOutcome.REJECT_UNSAFE
    assert decision.reason == "existing evaluator rejection: formation_desync"
    assert decision.trusted_checkpoint_allowed is False


def test_turn_clearance_translation_is_setup_even_when_it_crosses_route_progress():
    contract = {
        "active": True,
        "scope_checkpoint_id": "accepted_step_18",
        "preserve_turn_direction": "counterclockwise",
        "allowed_translation_directions": ["backward", "left"],
        "setup_only": True,
        "phase_completion_authorized": False,
    }
    decision = verify_transition(
        task=_task(),
        report=_report(1.2, 2.4),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "left", "coordination_mode": "sync"},
            verification_contract={"turn_clearance_replan": contract},
        ),
        before_state={"payload": {"active": True, "x": 0.0, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 1.2, "y": 2.4}},
        accepted_manifest={"steps": []},
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.positive_route_memory_allowed is False
    assert decision.evidence["turn_clearance_setup"]["active"] is True


def test_route_clearance_translation_is_setup_and_not_positive_route_memory():
    contract = {
        "active": True,
        "scope_checkpoint_id": "accepted_step_09",
        "preserve_navigation_world_direction_xy": [0.0, 1.0],
        "safe_heading_correction_turn_directions": ["counterclockwise"],
        "route_reentry_translation_directions": [],
        "clearance_setup_translation_directions": ["backward", "right"],
        "setup_only": True,
        "phase_completion_authorized": False,
    }
    decision = verify_transition(
        task=_task(),
        report=_report(1.2, 2.4),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "right", "coordination_mode": "sync"},
            verification_contract={
                "committed_subtask": {
                    "primitive": "coordinated_linear_transport",
                    "semantic_objective": "right",
                },
                "route_clearance_conflict": contract,
            },
        ),
        before_state={"payload": {"active": True, "x": 0.0, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 1.2, "y": 2.4}},
        accepted_manifest={"steps": []},
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.positive_route_memory_allowed is False
    assert decision.evidence["route_clearance_setup"] == {
        "active": True,
        "scope_checkpoint_id": "accepted_step_09",
        "primitive": "coordinated_linear_transport",
        "semantic_objective": "right",
        "preserve_navigation_world_direction_xy": [0.0, 1.0],
        "setup_only": True,
        "phase_completion_authorized": False,
    }


def test_valid_pivot_local_correction_is_accepted_inside_anchor_envelope():
    manifest = {
        "initial_state": {"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
        "steps": [
            {
                "checkpoint_after": "accepted_step_01",
                "measured_outcome": {"payload_x_m": 1.2, "payload_y_m": 2.4},
            }
        ],
    }
    decision = verify_transition(
        task=_task(),
        report=_report(
            1.05,
            2.5,
            evidence={
                "local_navigation_transition": {
                    "semantic_heading_improved": True,
                }
            },
        ),
        trace=_trace("coordinated_pivot"),
        before_state={"payload": {"active": True, "x": 1.2, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 1.05, "y": 2.5, "yaw_deg": 15.0}},
        accepted_manifest=manifest,
    )
    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.positive_route_memory_allowed is False


def test_pivot_drift_across_topology_gate_remains_setup_not_progress():
    decision = verify_transition(
        task=_task(),
        report=_report(
            1.2,
            2.4,
            evidence={
                "local_navigation_transition": {
                    "semantic_heading_improved": True,
                }
            },
        ),
        trace=_trace("coordinated_pivot"),
        before_state={"payload": {"active": True, "x": 1.1, "y": 2.4}},
        after_state={
            "payload": {"active": True, "x": 1.2, "y": 2.4, "yaw_deg": 15.0}
        },
        accepted_manifest={
            "initial_checkpoint": "accepted_step_00",
            "initial_state": {
                "payload": {"x": 1.1, "y": 2.4, "yaw_deg": 0.0}
            },
            "steps": [],
        },
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.positive_route_memory_allowed is False
    assert decision.evidence["route_progress"]["raw_topology_crossing"] is True
    assert decision.evidence["route_progress"]["topology_advanced"] is False
    assert decision.evidence["route_progress"]["completed_prefix_after"] == 0


def test_valid_alignment_correction_is_accepted_inside_anchor_envelope():
    manifest = {
        "initial_state": {"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
        "steps": [
            {
                "checkpoint_after": "accepted_step_01",
                "measured_outcome": {"payload_x_m": 1.2, "payload_y_m": 2.4},
            }
        ],
    }
    decision = verify_transition(
        task=_task(),
        report=_report(
            1.1,
            2.45,
            evidence={
                "alignment_progress_verification": {
                    "semantic_alignment_improved": True,
                }
            },
        ),
        trace=_trace("coordinated_alignment"),
        before_state={"payload": {"active": True, "x": 1.2, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 1.1, "y": 2.45}},
        accepted_manifest=manifest,
    )
    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.evidence["local_correction_envelope_applies"] is True


def test_lifecycle_transition_is_setup_until_terminal_contract_is_complete():
    setup = verify_transition(
        task=_task(),
        report=_report(0.0, 2.4),
        trace=_trace("place_payload_on_arms"),
        before_state={"payload": {"active": False, "x": 0.0, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 0.0, "y": 2.4}},
        accepted_manifest={"steps": []},
    )
    complete = verify_transition(
        task=_task(),
        report=EvaluationReport(
            success=True,
            failure_type="none",
            metrics={"task_complete": 1.0},
        ),
        trace=_trace("recover_stand"),
        before_state={"payload": {"active": True, "x": -1.5, "y": 7.6}},
        after_state={"payload": {"active": True, "x": -1.5, "y": 7.6}},
        accepted_manifest={"steps": []},
    )

    assert setup.outcome is TransitionOutcome.ACCEPT_SETUP
    assert not setup.positive_route_memory_allowed
    assert complete.outcome is TransitionOutcome.TASK_COMPLETE
    assert complete.positive_route_memory_allowed


def test_opposite_pivot_cannot_cancel_locked_accepted_turn_episode():
    before = {
        "payload": {"active": True, "x": 1.7, "y": 7.8},
        "turn_episode": {
            "active": True,
            "purpose": "route_turn",
            "direction": "counterclockwise",
            "switch_allowed": False,
            "latest_checkpoint_id": "accepted_step_37",
        },
    }
    decision = verify_transition(
        task=_task(),
        report=_report(1.8, 7.8),
        trace=_trace(
            "coordinated_pivot",
            {"turn_direction": "clockwise"},
        ),
        before_state=before,
        after_state={"payload": {"active": True, "x": 1.8, "y": 7.8}},
        accepted_manifest={"steps": []},
    )

    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.evidence["source"] == "turn_episode_continuity"
    assert decision.evidence["expected_turn_direction"] == "counterclockwise"


def test_opposite_pivot_is_allowed_after_turn_episode_switch_authorization():
    before = {
        "payload": {"active": True, "x": 0.9, "y": 2.4},
        "turn_episode": {
            "active": True,
            "purpose": "route_turn",
            "direction": "counterclockwise",
            "switch_allowed": True,
        },
    }
    decision = verify_transition(
        task=_task(),
        report=_report(
            0.9,
            2.4,
            evidence={
                "local_navigation_transition": {
                    "semantic_heading_improved": True,
                }
            },
        ),
        trace=_trace(
            "coordinated_pivot",
            {"turn_direction": "clockwise"},
        ),
        before_state=before,
        after_state={"payload": {"active": True, "x": 0.9, "y": 2.4}},
        accepted_manifest={"steps": []},
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP


def test_safe_local_correction_without_improvement_is_rejected():
    decision = verify_transition(
        task=_task(),
        report=_report(0.9, 2.4),
        trace=_trace("coordinated_pivot"),
        before_state={"payload": {"active": True, "x": 0.9, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 0.9, "y": 2.4}},
        accepted_manifest={"steps": []},
    )

    assert decision.outcome is TransitionOutcome.REJECT_REGRESSION
    assert decision.evidence["source"] == "local_setup_effect"


def test_corridor_recovery_is_setup_but_worsening_is_rejected():
    recovering = verify_transition(
        task=_task(),
        report=_report(0.2, 3.9),
        trace=_trace("coordinated_linear_transport"),
        before_state={"payload": {"active": True, "x": 0.0, "y": 4.2}},
        after_state={"payload": {"active": True, "x": 0.2, "y": 3.9}},
        accepted_manifest={"steps": []},
    )
    worsening = verify_transition(
        task=_task(),
        report=_report(0.2, 4.5),
        trace=_trace("coordinated_linear_transport"),
        before_state={"payload": {"active": True, "x": 0.0, "y": 4.2}},
        after_state={"payload": {"active": True, "x": 0.2, "y": 4.5}},
        accepted_manifest={"steps": []},
    )

    assert recovering.outcome is TransitionOutcome.ACCEPT_SETUP
    assert recovering.evidence["corridor_recovery"]["active"]
    assert worsening.outcome is TransitionOutcome.REJECT_REGRESSION
    assert worsening.evidence["source"] == "route_corridor_envelope"


def test_corridor_recovery_that_crosses_topology_gate_remains_progress():
    decision = verify_transition(
        task=_task(),
        report=_report(1.2, 2.4),
        trace=_trace("coordinated_linear_transport"),
        before_state={"payload": {"active": True, "x": 0.0, "y": 4.2}},
        after_state={"payload": {"active": True, "x": 1.2, "y": 2.4}},
        accepted_manifest={"steps": []},
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_PROGRESS
    assert decision.evidence["corridor_recovery"]["active"]
    assert decision.evidence["route_progress"]["topology_advanced"]


def test_latest_phase_anchor_supersedes_earlier_axis_after_second_corner():
    phase_guard = {
        "maximum_completed_prefix": 3,
        "phase_anchors": [
            {
                "completed_prefix": 1,
                "payload_x_m": 1.407718,
                "payload_y_m": 2.345883,
                "checkpoint_after": "accepted_step_06",
            },
            {
                "completed_prefix": 3,
                "payload_x_m": 1.52096,
                "payload_y_m": 7.607612,
                "checkpoint_after": "accepted_step_33",
            },
        ],
    }

    retained = _phase_anchor_regression(
        task=_task(),
        after_state={"payload": {"x": 0.787248, "y": 7.364609}},
        phase_guard=phase_guard,
    )
    regressed = _phase_anchor_regression(
        task=_task(),
        after_state={"payload": {"x": 0.787248, "y": 6.90}},
        phase_guard=phase_guard,
    )

    assert retained == {}
    assert regressed["violated"] is True
    assert regressed["completed_prefix"] == 3


def test_phase_anchor_retention_is_equivariant_for_rotated_scene_gates():
    task = BoxCarryTask(
        task_id="rotated_route",
        target="place on target",
        route_progress_gates=(
            RouteProgressGate("gate_a", 0.0, 1.0, 2.0),
            RouteProgressGate("gate_b", -1.0, 0.0, -3.0),
        ),
    )
    phase_guard = {
        "maximum_completed_prefix": 1,
        "phase_anchors": [
            {
                "completed_prefix": 1,
                "payload_x_m": 4.0,
                "payload_y_m": 2.2,
                "checkpoint_after": "accepted_step_04",
            }
        ],
    }

    assert _phase_anchor_regression(
        task=task,
        after_state={"payload": {"x": -50.0, "y": 2.0}},
        phase_guard=phase_guard,
    ) == {}
    regressed = _phase_anchor_regression(
        task=task,
        after_state={"payload": {"x": 50.0, "y": 1.5}},
        phase_guard=phase_guard,
    )
    assert regressed["violated"] is True


def test_completed_gate_retention_does_not_freeze_room_scale_overshoot():
    phase_guard = {
        "maximum_completed_prefix": 3,
        "phase_anchors": [
            {
                "completed_prefix": 3,
                "payload_x_m": 1.899657,
                "payload_y_m": 7.938326,
                "checkpoint_after": "accepted_step_16",
            }
        ],
    }

    # V31 accepted a long cross-passage action well beyond the second-corner
    # gate at y=7.35. The next valid -X segment can lower y slightly while still
    # remaining on the completed side of that authored topology boundary.
    retained = _phase_anchor_regression(
        task=_task(),
        after_state={
            "payload": {"x": 1.1026346313804503, "y": 7.460893872581031}
        },
        phase_guard=phase_guard,
    )
    regressed = _phase_anchor_regression(
        task=_task(),
        after_state={"payload": {"x": 1.10, "y": 6.90}},
        phase_guard=phase_guard,
    )

    assert retained == {}
    assert regressed["violated"] is True
    assert regressed["completed_gate_threshold_m"] == pytest.approx(7.35)
    assert regressed["candidate_gate_progress_m"] == pytest.approx(6.90)


def test_v31_second_corner_route_aligned_transport_is_restorable_setup():
    manifest = {
        "initial_state": {"payload": {"x": -1.2, "y": 2.4}},
        "steps": [
            {
                "checkpoint_after": "accepted_step_04",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {
                    "payload_x_m": 2.487949,
                    "payload_y_m": 2.390648,
                },
            },
            {
                "checkpoint_after": "accepted_step_13",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {
                    "payload_x_m": 2.402693,
                    "payload_y_m": 3.974884,
                },
            },
            {
                "checkpoint_after": "accepted_step_16",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {
                    "payload_x_m": 1.899657,
                    "payload_y_m": 7.938326,
                },
            },
        ],
    }
    verification_contract = {
        "committed_subtask": {
            "primitive": "coordinated_linear_transport",
            "semantic_objective": "left",
        },
        "route_clearance_conflict": {
            "active": True,
            "scope_checkpoint_id": "accepted_step_19",
            "preserve_navigation_world_direction_xy": [-1.0, 0.0],
            "route_reentry_translation_directions": ["left"],
            "clearance_setup_translation_directions": [
                "forward",
                "backward",
                "right",
            ],
            "safe_heading_correction_turn_directions": ["counterclockwise"],
            "setup_only": True,
            "phase_completion_authorized": False,
        },
    }

    decision = verify_transition(
        task=_task(),
        report=_report(1.105729272184606, 7.462072329585735),
        trace=_trace(
            "coordinated_linear_transport",
            {"direction": "left"},
            verification_contract=verification_contract,
        ),
        before_state={
            "payload": {"active": True, "x": 1.343679, "y": 7.671672}
        },
        after_state={
            "payload": {
                "active": True,
                "x": 1.105729272184606,
                "y": 7.462072329585735,
            }
        },
        accepted_manifest=manifest,
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.trusted_checkpoint_allowed is True
    assert decision.evidence["active_phase_progress"]["delta_m"] > 0.0
    assert decision.evidence["route_segment_after"]["inside_corridor_envelope"]


def test_terminal_anchor_regression_uses_target_distance_without_gate_fields():
    task = _task()
    phase_guard = {
        "maximum_completed_prefix": len(task.route_progress_gates) + 1,
        "phase_anchors": [
            {
                "completed_prefix": len(task.route_progress_gates) + 1,
                "payload_x_m": task.target_center_x_m,
                "payload_y_m": task.target_center_y_m,
                "checkpoint_after": "accepted_step_terminal",
            }
        ],
    }

    regression = _phase_anchor_regression(
        task=task,
        after_state={
            "payload": {
                "x": task.target_center_x_m + 1.5,
                "y": task.target_center_y_m,
            }
        },
        phase_guard=phase_guard,
    )

    assert regression["violated"] is True
    assert "completed_gate_threshold_m" not in regression
    assert "candidate_gate_progress_m" not in regression


def test_route_reference_validation_is_rotation_equivariant_and_phase_name_neutral():
    base_plan = {
        "current_subtask": {
            "strategy": {"primitive": "coordinated_linear_transport"},
            "navigation": {
                "reference": "visible_connected_free_space",
                "world_direction_xy": [1.0, 0.0],
            },
        }
    }
    base_task = replace(
        _task(),
        route_progress_gates=(RouteProgressGate("arbitrary_name", 1.0, 0.0, 1.0),),
        route_segments=(
            RouteSegment(
                "segment_alpha",
                1.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                1.0,
                0.0,
                0.5,
            ),
        ),
    )
    base = validate_task_plan_route_reference(
        base_task,
        {"payload": {"x": 0.25, "y": 0.0}},
        completed_prefix=0,
        task_plan=base_plan,
    )

    rotated_plan = {
        "current_subtask": {
            "strategy": {"primitive": "coordinated_linear_transport"},
            "navigation": {
                "reference": "visible_connected_free_space",
                "world_direction_xy": [0.0, 1.0],
            },
        }
    }
    rotated_task = replace(
        base_task,
        route_progress_gates=(RouteProgressGate("renamed_phase", 0.0, 1.0, 1.0),),
        route_segments=(
            RouteSegment(
                "renamed_segment",
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                1.0,
                0.5,
            ),
        ),
    )
    rotated = validate_task_plan_route_reference(
        rotated_task,
        {"payload": {"x": 0.0, "y": 0.25}},
        completed_prefix=0,
        task_plan=rotated_plan,
    )

    assert base["valid"] and rotated["valid"]
    assert base["direction_dot"] == rotated["direction_dot"] == 1.0


def test_wrong_route_direction_is_rejected_and_locked_reference_survives_mutation():
    task = replace(
        _task(),
        route_progress_gates=(RouteProgressGate("gate", 1.0, 0.0, 1.0),),
        route_segments=(
            RouteSegment(
                "connected_segment",
                1.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                1.0,
                0.0,
                0.5,
            ),
        ),
    )
    wrong = {
        "current_subtask": {
            "strategy": {"primitive": "coordinated_pivot"},
            "navigation": {
                "reference": "visible_connected_free_space",
                "world_direction_xy": [-1.0, 0.0],
            },
        }
    }
    rejected = validate_task_plan_route_reference(
        task,
        {"payload": {"x": 0.25, "y": 0.0}},
        completed_prefix=0,
        task_plan=wrong,
    )
    assert rejected["active"] and not rejected["valid"]
    assert "contradicts" in rejected["reason"]

    locked = {
        "current_subtask": {
            "strategy": {"primitive": "coordinated_linear_transport"},
            "navigation": {
                "reference": "visible_connected_free_space",
                "world_direction_xy": [1.0, 0.0],
            },
        }
    }
    assert validate_task_plan_route_reference(
        task,
        {"payload": {"x": 0.25, "y": 0.0}},
        completed_prefix=0,
        task_plan=locked,
    )["valid"]
    mutated = {
        "current_subtask": {
            "strategy": {"primitive": "coordinated_pivot"},
            "navigation": {
                **locked["current_subtask"]["navigation"],
                "world_direction_xy": [0.0, 1.0],
            },
        }
    }
    lock_check = validate_locked_route_reference_unchanged(locked, mutated)
    assert lock_check["active"] and not lock_check["valid"]


def test_turn_budget_replan_requires_newer_global_observation():
    task = replace(
        _task(),
        route_progress_gates=(RouteProgressGate("gate", 1.0, 0.0, 1.0),),
        route_segments=(
            RouteSegment(
                "connected_segment",
                1.0,
                0.0,
                0.0,
                0.0,
                2.0,
                0.0,
                1.0,
                0.0,
                0.5,
            ),
        ),
    )

    def plan() -> dict[str, object]:
        return {
            "current_subtask": {
                "strategy": {"primitive": "coordinated_pivot"},
                "navigation": {
                    "reference": "visible_connected_free_space",
                    "world_direction_xy": [1.0, 0.0],
                },
            }
        }

    exhausted = {
        "payload": {"x": 0.25, "y": 0.0},
        "turn_episode": {
            "status": "local_direction_replan_required",
            "replan_requires_fresh_global_observation": True,
            "route_observation_captured_at_unix": 10.0,
        },
        "global_observation": {
            "available": True,
            "captured_at_unix": 10.0,
            "source": "global",
        },
    }
    stale = validate_task_plan_route_reference(
        task,
        exhausted,
        completed_prefix=0,
        task_plan=plan(),
    )
    fresh_state = {
        **exhausted,
        "global_observation": {
            "available": True,
            "captured_at_unix": 11.0,
            "source": "global",
        },
    }
    fresh_plan = plan()
    fresh = validate_task_plan_route_reference(
        task,
        fresh_state,
        completed_prefix=0,
        task_plan=fresh_plan,
    )

    assert stale["active"] and not stale["valid"]
    assert "newer global observation" in stale["reason"]
    assert fresh["valid"]
    assert fresh_plan["current_subtask"]["navigation"][
        "turn_budget_replan_validated"
    ]


def test_action_history_alone_cannot_complete_a_phase():
    phases = [
        {"phase_id": "initial straight", "abstract_goal": "initial straight"},
        {"phase_id": "first corner", "abstract_goal": "first corner"},
    ]
    evidence = _accepted_route_phase_evidence(
        phases,
        context={
            "accepted_route_action_history": {
                "steps": [
                    {
                        "method": "coordinated_linear_transport",
                        "transition_outcome": "ACCEPT_PROGRESS",
                        "positive_route_memory_eligible": True,
                    },
                    {
                        "method": "coordinated_pivot",
                        "transition_outcome": "ACCEPT_PROGRESS",
                        "positive_route_memory_eligible": True,
                    },
                ]
            },
            "host_route_phase_guard": {"maximum_completed_prefix": 0},
        },
    )
    assert evidence["action_evidence_maximum_prefix"] == 2
    assert evidence["maximum_completed_prefix"] == 0


def test_ffbbff_state_cycle_is_detected():
    points = [0.5, 1.0, 0.5, 0.0, 0.5]
    manifest = {
        "initial_state": {"payload": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}},
        "steps": [
            {
                "measured_outcome": {
                    "payload_x_m": x,
                    "payload_y_m": 0.0,
                    "payload_step_signed_yaw_change_deg": 0.0,
                }
            }
            for x in points
        ],
    }
    cycle = detect_state_progress_cycle(
        task=_task(),
        accepted_manifest=manifest,
        candidate_report=_report(1.0, 0.0),
        candidate_state={"payload": {"x": 1.0, "y": 0.0, "yaw_deg": 0.0}},
    )
    assert cycle["detected"] is True
    assert cycle["kind"] == "coarse_state_recurrence"
    assert cycle["departed_before_return"] is True
    assert cycle["cycle_semantics_version"] == 2


def test_setup_checkpoints_are_route_cycle_trajectory_evidence():
    manifest = {
        "initial_state": {"payload": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}},
        "steps": [
            {
                "transition_outcome": "ACCEPT_SETUP",
                "positive_route_memory_eligible": False,
                "measured_outcome": {
                    "payload_x_m": x,
                    "payload_y_m": 0.0,
                    "payload_step_signed_yaw_change_deg": 0.0,
                },
            }
            for x in (0.5, 1.0, 0.5, 0.0, 0.5)
        ],
    }

    cycle = detect_state_progress_cycle(
        task=_task(),
        accepted_manifest=manifest,
        candidate_report=_report(1.0, 0.0),
        candidate_state={"payload": {"x": 1.0, "y": 0.0, "yaw_deg": 0.0}},
    )

    assert cycle["detected"] is True
    assert cycle["kind"] == "coarse_state_recurrence"


def test_cycle_signature_distinguishes_deterministic_formation_repair():
    def setup_step(x: float, stagger_m: float) -> dict[str, object]:
        return {
            "transition_outcome": "ACCEPT_SETUP",
            "measured_outcome": {
                "payload_x_m": x,
                "payload_y_m": 2.4,
                "payload_step_signed_yaw_change_deg": 0.0,
                "robot_pair_longitudinal_stagger_m": stagger_m,
                "robot_pair_formation_skew_deg": 0.0,
                "robot_heading_disagreement_after_deg": 0.0,
                "payload_support_center_error_m": 0.0,
            },
        }

    manifest = {
        "initial_state": {"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
        "steps": [
            setup_step(0.0, 0.20),
            setup_step(-0.5, 0.20),
            setup_step(-0.5, 0.0),
        ],
    }
    repaired = detect_state_progress_cycle(
        task=_task(),
        accepted_manifest=manifest,
        candidate_report=EvaluationReport(
            success=True,
            failure_type="none",
            metrics={
                "payload_x_m": 0.0,
                "payload_y_m": 2.4,
                "robot_pair_longitudinal_stagger_m": 0.0,
                "robot_pair_formation_skew_deg": 0.0,
                "robot_heading_disagreement_after_deg": 0.0,
                "payload_support_center_error_m": 0.0,
            },
        ),
        candidate_state={"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
        candidate_phase_prefix=0,
    )
    unrepaired = detect_state_progress_cycle(
        task=_task(),
        accepted_manifest=manifest,
        candidate_report=EvaluationReport(
            success=True,
            failure_type="none",
            metrics={
                "payload_x_m": 0.0,
                "payload_y_m": 2.4,
                "robot_pair_longitudinal_stagger_m": 0.20,
                "robot_pair_formation_skew_deg": 0.0,
                "robot_heading_disagreement_after_deg": 0.0,
                "payload_support_center_error_m": 0.0,
            },
        ),
        candidate_state={"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
        candidate_phase_prefix=0,
    )

    assert repaired["detected"] is False
    assert unrepaired["detected"] is True


def test_deterministically_improving_local_repair_is_not_cycle_rejected():
    manifest = {
        "initial_state": {"payload": {"x": 0.0, "y": 2.4, "yaw_deg": 0.0}},
        "steps": [
            {
                "transition_outcome": "ACCEPT_SETUP",
                "measured_outcome": {
                    "payload_x_m": -0.5,
                    "payload_y_m": 2.4,
                    "payload_step_signed_yaw_change_deg": 0.0,
                },
            },
            {
                "transition_outcome": "ACCEPT_SETUP",
                "measured_outcome": {
                    "payload_x_m": -0.5,
                    "payload_y_m": 2.4,
                    "payload_step_signed_yaw_change_deg": 0.0,
                },
            },
        ],
    }
    decision = verify_transition(
        task=_task(),
        report=_report(
            0.0,
            2.4,
            evidence={
                "alignment_progress_verification": {
                    "semantic_alignment_improved": True,
                }
            },
        ),
        trace=_trace("coordinated_alignment"),
        before_state={"payload": {"active": True, "x": -0.5, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 0.0, "y": 2.4}},
        accepted_manifest=manifest,
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.evidence["local_setup"]["formation_improved"] is True


def test_v70_conflicting_route_scalars_do_not_create_positive_memory():
    manifest = {
        "initial_state": {
            "payload": {"x": -1.2, "y": 2.4, "yaw_deg": 0.0}
        },
        "steps": [
            {
                "checkpoint_after": "accepted_step_03",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {
                    "payload_x_m": 1.26697,
                    "payload_y_m": 2.411725,
                },
            }
        ],
    }
    decision = verify_transition(
        task=_task(),
        report=_report(2.578181, 2.471741),
        trace=_trace("coordinated_linear_transport", {"direction": "forward"}),
        before_state={
            "payload": {"active": True, "x": 1.26926, "y": 2.412446}
        },
        after_state={
            "payload": {"active": True, "x": 2.578181, "y": 2.471741}
        },
        accepted_manifest=manifest,
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_SETUP
    assert decision.positive_route_memory_allowed is False
    assert decision.evidence["route_progress"]["along_segment_progress_m"] > 0.05
    assert decision.evidence["route_progress"]["local_goal_progress_m"] < -0.8
    assert decision.evidence["route_progress"]["consistent_segment_progress"] is False


def test_v70_state_beyond_future_gates_cannot_jump_topology_prefix():
    manifest = {
        "initial_state": {
            "payload": {"x": -1.2, "y": 2.4, "yaw_deg": 0.0}
        },
        "steps": [
            {
                "checkpoint_after": "accepted_step_03",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {
                    "payload_x_m": 1.26697,
                    "payload_y_m": 2.411725,
                },
            },
            {
                "checkpoint_after": "accepted_step_21",
                "transition_outcome": "ACCEPT_SETUP",
                "positive_route_memory_eligible": False,
                "measured_outcome": {
                    "payload_x_m": 1.037669,
                    "payload_y_m": 7.736511,
                },
            },
        ],
    }
    decision = verify_transition(
        task=_task(),
        report=_report(1.775514, 7.677422),
        trace=_trace("coordinated_linear_transport", {"direction": "forward"}),
        before_state={
            "payload": {"active": True, "x": 1.037669, "y": 7.736511}
        },
        after_state={
            "payload": {"active": True, "x": 1.775514, "y": 7.677422}
        },
        accepted_manifest=manifest,
    )

    assert decision.outcome is not TransitionOutcome.ACCEPT_PROGRESS
    assert decision.positive_route_memory_allowed is False
    assert decision.evidence["raw_completed_prefix_after"] == 2
    assert decision.evidence["route_progress"]["completed_prefix_after"] == 1
    assert decision.evidence["active_gate_crossed"] is False


def test_one_transaction_advances_at_most_the_active_topology_gate():
    decision = verify_transition(
        task=_task(),
        report=_report(2.0, 7.5),
        trace=_trace("coordinated_linear_transport", {"direction": "forward"}),
        before_state={"payload": {"active": True, "x": 1.0, "y": 2.4}},
        after_state={"payload": {"active": True, "x": 2.0, "y": 7.5}},
        accepted_manifest={
            "initial_checkpoint": "accepted_step_00",
            "initial_state": {
                "payload": {"x": 1.0, "y": 2.4, "yaw_deg": 0.0}
            },
            "steps": [],
        },
    )

    assert decision.outcome is TransitionOutcome.ACCEPT_PROGRESS
    assert decision.evidence["raw_completed_prefix_after"] == 1
    assert decision.evidence["route_progress"]["completed_prefix_after"] == 1

    persisted = trusted_route_phase_guard(
        _task(),
        {"payload": {"active": True, "x": 2.0, "y": 7.5}},
        accepted_steps=[
            {
                "checkpoint_after": "accepted_step_01",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_eligible": True,
                "measured_outcome": {"payload_x_m": 2.0, "payload_y_m": 7.5},
            }
        ],
    )
    assert persisted["maximum_completed_prefix"] == 1
    assert [
        item["completed_prefix"] for item in persisted["phase_anchors"]
    ] == [1]


def test_sequential_topology_preserves_v57_success_route_endpoints():
    # Accepted progress endpoints from V57 Claude seed_001. Intervening setup
    # checkpoints are intentionally absent from route-completion chronology.
    endpoints = [
        (1.53, 2.39),
        (1.93, 2.95),
        (1.98, 4.25),
        (2.41, 7.93),
        (0.23, 7.81),
        (-0.68, 7.54),
        (-1.54, 7.55),
    ]
    steps = [
        {
            "checkpoint_after": f"accepted_step_{index:02d}",
            "transition_outcome": "ACCEPT_PROGRESS",
            "positive_route_memory_eligible": True,
            "measured_outcome": {"payload_x_m": x, "payload_y_m": y},
        }
        for index, (x, y) in enumerate(endpoints, 1)
    ]
    guard = trusted_route_phase_guard(
        _task(),
        {
            "payload": {
                "active": True,
                "x": endpoints[-1][0],
                "y": endpoints[-1][1],
            }
        },
        accepted_steps=steps,
    )

    assert guard["maximum_completed_prefix"] == 5
    assert [item["completed_prefix"] for item in guard["phase_anchors"]] == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_dense_progress_inside_one_coarse_cell_is_not_a_cycle():
    points = [(-0.2452, 7.2897), (-0.2673, 7.3290), (-0.2604, 7.3365)]
    manifest = {
        "initial_state": {
            "payload": {"x": -0.2141, "y": 6.9359, "yaw_deg": 207.9}
        },
        "steps": [
            {
                "measured_outcome": {
                    "payload_x_m": x,
                    "payload_y_m": y,
                    "payload_step_signed_yaw_change_deg": yaw_delta,
                }
            }
            for (x, y), yaw_delta in zip(points, (-22.1, -2.0, 1.5))
        ],
    }
    cycle = detect_state_progress_cycle(
        task=_task(),
        accepted_manifest=manifest,
        candidate_report=_report(-0.3308, 7.3015),
        candidate_state={
            "payload": {"x": -0.3308, "y": 7.3015, "yaw_deg": 175.4}
        },
    )

    assert cycle["detected"] is False


def test_repeated_semantic_pivot_failure_forces_structural_mutation(tmp_path):
    def policy(attempt: int, angle_deg: float) -> PolicyScript:
        scripts = []
        for robot_id, role in (("front", "inner"), ("back", "outer")):
            code = (
                "def policy(api):\n"
                "    api.coordinated_pivot("
                "turn_direction='clockwise', "
                f"angle_deg={angle_deg}, travel_direction='forward', duration_s=4.0, "
                f"role='{role}', swing_radius_m=0.5)\n"
                "    api.stop_all()\n"
            )
            scripts.append(
                RobotPolicyScript(
                    robot_id=robot_id,
                    script_id=f"{robot_id}_{attempt}",
                    code=code,
                    synchronization_group="team",
                    coordination_mode="async",
                )
            )
        return PolicyScript(
            script_id=f"pivot_{attempt}",
            attempt=attempt,
            code=scripts[0].code,
            robot_scripts=tuple(scripts),
        )

    controller = EvolutionController(output_dir=tmp_path, resume_history=False)
    failure = EvaluationReport(
        success=False,
        failure_type="payload_unstable",
        failed_subtask="coordinated_pivot",
    )
    first = controller.record_and_decide(
        policy=policy(1, 30.0),
        report=failure,
        trace=None,
        checkpoint_id="accepted_step_03",
        physically_dispatched=True,
    )
    second = controller.record_and_decide(
        policy=policy(2, 20.0),
        report=failure,
        trace=None,
        checkpoint_id="accepted_step_03",
        physically_dispatched=True,
    )

    assert first.semantic_failure_count == 1
    assert second.semantic_failure_count == 2
    assert second.repair_level in {"coordination", "strategy"}
    assert second.action != "tune_parameter"
    assert "parameter_tuning is forbidden" in second.required_change


def test_setup_checkpoint_does_not_reset_route_stall_evidence(tmp_path):
    policy = PolicyScript(
        script_id="setup",
        attempt=1,
        code="def policy(api):\n    api.stop_all()\n",
    )
    controller = EvolutionController(output_dir=tmp_path, resume_history=False)

    first = controller.record_and_decide(
        policy=policy,
        report=_annotated_report(TransitionOutcome.ACCEPT_SETUP),
        trace=None,
        checkpoint_id="accepted_step_00",
        physically_dispatched=True,
    )
    second = controller.record_and_decide(
        policy=replace(policy, attempt=2, script_id="setup_2"),
        report=_annotated_report(TransitionOutcome.ACCEPT_SETUP),
        trace=None,
        checkpoint_id="accepted_step_01",
        physically_dispatched=True,
    )
    progress = controller.record_and_decide(
        policy=replace(policy, attempt=3, script_id="progress"),
        report=_annotated_report(TransitionOutcome.ACCEPT_PROGRESS),
        trace=None,
        checkpoint_id="accepted_step_02",
        physically_dispatched=True,
    )

    assert first.route_stall_iterations == 1
    assert second.route_stall_iterations == 2
    assert progress.route_stall_iterations == 0
    assert controller.metrics()["maximum_route_stall_iterations"] == 2


def test_setup_checkpoint_is_restorable_but_cannot_advance_topology():
    task = _task()
    progress_step = {
        "transition_outcome": "ACCEPT_PROGRESS",
        "positive_route_memory_eligible": True,
        "measured_outcome": {"payload_x_m": 1.2, "payload_y_m": 2.35},
    }
    setup_step = {
        "transition_outcome": "ACCEPT_SETUP",
        "positive_route_memory_eligible": False,
        "measured_outcome": {"payload_x_m": 1.35, "payload_y_m": 3.2},
    }
    setup_state = {
        "payload": {"active": True, "x": 1.35, "y": 3.2},
        "checkpoint": {"checkpoint_id": "accepted_step_02"},
    }

    guard = trusted_route_phase_guard(
        task,
        setup_state,
        accepted_steps=[progress_step, setup_step],
    )
    candidate_guard = trusted_route_phase_guard(
        task,
        setup_state,
        accepted_steps=[progress_step],
        include_candidate_state=True,
    )

    assert guard["current_state_route_eligible"] is False
    assert guard["maximum_completed_prefix"] == 1
    assert candidate_guard["maximum_completed_prefix"] == 2


def test_trusted_route_phase_guard_is_total_after_terminal_stand_recovery():
    task = _task()
    guard = trusted_route_phase_guard(
        task,
        {
            "payload": {"active": True, "x": -1.5, "y": 7.6},
            "post_release_lifecycle": {"stand_recovered": True},
        },
        accepted_steps=[],
    )

    assert guard["current_topological_phase"] == "mission_complete"
    assert guard["maximum_completed_prefix"] == len(task.route_progress_gates) + 2
    assert guard["current_state_route_eligible"] is False
    assert guard["phase_anchors"] == []


def test_post_transition_checkpoint_failure_overrides_embedded_accept_outcome():
    report = _annotated_report(TransitionOutcome.ACCEPT_SETUP)
    invalidated = replace(
        report,
        success=False,
        failure_type="payload_unstable",
    )

    assert transition_outcome_from_report(invalidated) is (
        TransitionOutcome.REJECT_UNSAFE
    )
