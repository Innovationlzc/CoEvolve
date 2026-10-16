from __future__ import annotations

from dataclasses import replace
import json
from math import radians
import time

import pytest

from praxis_coop.cap_policy.contracts import PolicyScript, RobotPolicyScript
from praxis_coop.cap_policy.coordination_specs import (
    coordination_primitive_method_spec,
    validate_coordination_primitive_parameters,
)
from praxis_coop.cap_policy.evolution_controller import (
    policy_role_signature,
    policy_strategy_signature,
)
from praxis_coop.cap_policy.policy_executor import validate_team_policy_scripts
from praxis_coop.cap_policy.strategy_memory import StrategyMemoryStore
from praxis_coop.sonic.team_coordination import (
    CoordinatedTeamPlan,
    build_coordinated_schedule,
    schedule_duration,
)
from praxis_coop.sonic.coordination_primitives import (
    SONIC_ALIGNMENT_TRANSLATION_COMMAND_MPS,
    SONIC_ARC_TRANSLATION_ONSET_MPS,
    calibrated_alignment_translation_speed,
)
from praxis_coop.sonic.alignment_feedback import RelativeAlignmentFeedback
from praxis_coop.sonic.turn_feedback import PayloadYawFeedback


def _arc_parameters(radius_m: float) -> dict[str, object]:
    return {
        "turn_direction": "counterclockwise",
        "angle_deg": 15.0,
        "duration_s": 4.0,
        "radius_m": radius_m,
    }


def _arc_code(radius_m: float) -> str:
    return (
        "def policy(api):\n"
        "    api.coordinated_arc_transport("
        'turn_direction="counterclockwise", angle_deg=15.0, '
        f"duration_s=4.0, radius_m={radius_m})\n"
        "    api.stop_all()\n"
    )


def test_loaded_turn_specs_allow_thirty_degree_route_segments():
    for method in (
        "coordinated_arc_transport",
        "coordinated_pivot",
    ):
        angle = coordination_primitive_method_spec(method)["parameters"]["angle_deg"]
        assert angle["min"] == pytest.approx(2.0)
        assert angle["default"] == pytest.approx(30.0)
        assert angle["max"] == pytest.approx(30.0)


def test_payload_yaw_feedback_stops_at_measured_target_with_wraparound(tmp_path):
    state_file = tmp_path / "runtime_state.json"
    state_file.write_text(
        json.dumps(
            {
                "time_unix": time.time(),
                "payload": {"yaw_rad": radians(179.0)},
            }
        ),
        encoding="utf-8",
    )
    feedback = PayloadYawFeedback(
        state_file,
        expected_direction=1,
        stop_margin_rad=radians(0.5),
    )

    state_file.write_text(
        json.dumps(
            {
                "time_unix": time.time(),
                "payload": {"yaw_rad": radians(-176.6)},
            }
        ),
        encoding="utf-8",
    )

    assert feedback.reached(radians(5.0)) is False
    state_file.write_text(
        json.dumps(
            {
                "time_unix": time.time(),
                "payload": {"yaw_rad": radians(-176.4)},
            }
        ),
        encoding="utf-8",
    )
    assert feedback.reached(radians(5.0)) is True
    assert feedback.report()["max_progress_rad"] == pytest.approx(radians(4.6))


def _arc_scripts() -> tuple[RobotPolicyScript, ...]:
    return (
        RobotPolicyScript(
            robot_id="front",
            script_id="front_arc",
            code=_arc_code(0.8),
            synchronization_group="corner_arc",
            phase_id="corner_arc",
            contact_intent="maintain_payload_support",
            coordination_mode="async",
        ),
        RobotPolicyScript(
            robot_id="back",
            script_id="back_arc",
            code=_arc_code(0.2),
            synchronization_group="corner_arc",
            phase_id="corner_arc",
            contact_intent="maintain_payload_support",
            coordination_mode="async",
        ),
    )


def test_arc_transport_compiles_simultaneous_outer_and_inner_arcs():
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "async",
            "phase_id": "corner_arc",
            "synchronization_group": "corner_arc",
            "actions": {
                "front": {
                    "method": "coordinated_arc_transport",
                    "parameters": _arc_parameters(0.8),
                },
                "back": {
                    "method": "coordinated_arc_transport",
                    "parameters": _arc_parameters(0.2),
                },
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=2.0,
        transport_duration_gain=1.0,
        transport_duration_baseline_m=2.0,
        minimum_transport_ramp_s=1.5,
    )
    arc_groups = [
        group
        for group in groups
        if group["front"].motion_axis == "arc"
    ]

    assert arc_groups
    for group in arc_groups:
        outer = group["front"]
        inner = group["back"]
        assert outer.duration_s == pytest.approx(inner.duration_s)
        assert outer.delta_yaw_rad == pytest.approx(inner.delta_yaw_rad)
        assert outer.speed > SONIC_ARC_TRANSLATION_ONSET_MPS
        assert outer.movement[:2] == (1.0, 0.0)
        assert SONIC_ARC_TRANSLATION_ONSET_MPS < inner.speed < outer.speed
        assert inner.movement[:2] == (1.0, 0.0)


def test_arc_transport_uses_model_selected_body_lateral_direction():
    front = {**_arc_parameters(0.8), "travel_direction": "left"}
    back = {**_arc_parameters(0.2), "travel_direction": "left"}
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "async",
            "phase_id": "lateral_corner_arc",
            "synchronization_group": "lateral_corner_arc",
            "actions": {
                "front": {
                    "method": "coordinated_arc_transport",
                    "parameters": front,
                },
                "back": {
                    "method": "coordinated_arc_transport",
                    "parameters": back,
                },
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=2.0,
        transport_duration_gain=1.0,
        transport_duration_baseline_m=2.0,
        minimum_transport_ramp_s=1.5,
    )
    arc_groups = [
        group for group in groups if group["front"].motion_axis == "arc"
    ]

    assert arc_groups
    assert all(group["front"].movement[:2] == (0.0, 1.0) for group in arc_groups)
    assert all(group["back"].movement[:2] == (0.0, 1.0) for group in arc_groups)


def test_linear_transport_preserves_verified_sonic_body_left_sign():
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "sync",
            "phase_id": "body_left_transport",
            "synchronization_group": "body_left_transport",
            "actions": {
                robot_id: {
                    "method": "coordinated_linear_transport",
                    "parameters": {
                        "direction": "left",
                        "distance_m": 0.5,
                        "speed_mps": 0.5,
                        "ramp_s": 1.5,
                    },
                }
                for robot_id in ("front", "back")
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=0.4,
        transport_duration_gain=1.0,
        transport_duration_baseline_m=2.0,
        minimum_transport_ramp_s=1.5,
    )
    motion = next(
        group
        for group in groups
        if group["front"].motion_axis == "lateral"
        and group["front"].speed > 0.0
    )

    assert motion["front"].movement == motion["back"].movement == (0.0, 1.0, 0.0)
    assert "carry_lateral_left" in motion["front"].phase_id


def test_arc_transport_rejects_mismatched_body_travel_directions():
    front = {**_arc_parameters(0.8), "travel_direction": "left"}
    back = {**_arc_parameters(0.2), "travel_direction": "right"}

    with pytest.raises(ValueError, match="matching front/back travel_direction"):
        CoordinatedTeamPlan.from_payload(
            {
                "coordination_mode": "async",
                "phase_id": "invalid_arc",
                "synchronization_group": "invalid_arc",
                "actions": {
                    "front": {
                        "method": "coordinated_arc_transport",
                        "parameters": front,
                    },
                    "back": {
                        "method": "coordinated_arc_transport",
                        "parameters": back,
                    },
                },
            }
        )


def test_removed_payload_centered_in_place_turn_is_rejected():
    parameters = {
        "turn_direction": "counterclockwise",
        "angle_deg": 10.0,
        "duration_s": 6.0,
    }
    with pytest.raises(ValueError, match="unsupported coordinated action"):
        CoordinatedTeamPlan.from_payload(
            {
                "coordination_mode": "sync",
                "phase_id": "loaded_in_place_turn",
                "synchronization_group": "loaded_in_place_turn",
                "actions": {
                    robot_id: {
                        "method": "coordinated_in_place_turn",
                        "parameters": dict(parameters),
                    }
                    for robot_id in ("front", "back")
                },
            }
        )


def test_arc_transport_requires_async_distinct_radii():
    payload = {
        "coordination_mode": "async",
        "phase_id": "corner_arc",
        "synchronization_group": "corner_arc",
        "actions": {
            robot_id: {
                "method": "coordinated_arc_transport",
                "parameters": _arc_parameters(0.4),
            }
            for robot_id in ("front", "back")
        },
    }

    with pytest.raises(ValueError, match="distinct inner/outer"):
        CoordinatedTeamPlan.from_payload(payload)

    payload["actions"]["front"]["parameters"] = _arc_parameters(0.8)
    payload["coordination_mode"] = "sync"
    with pytest.raises(ValueError, match="requires coordination_mode=async"):
        CoordinatedTeamPlan.from_payload(payload)


def test_arc_transport_rejects_zero_radius_and_points_to_pivot():
    payload = {
        "coordination_mode": "async",
        "phase_id": "corner_arc",
        "synchronization_group": "corner_arc",
        "actions": {
            "front": {
                "method": "coordinated_arc_transport",
                "parameters": _arc_parameters(0.8),
            },
            "back": {
                "method": "coordinated_arc_transport",
                "parameters": _arc_parameters(0.0),
            },
        },
    }

    with pytest.raises(ValueError, match="use coordinated_pivot"):
        CoordinatedTeamPlan.from_payload(payload)


def test_policy_executor_keeps_one_call_boundary_for_arc_profiles():
    scripts = _arc_scripts()
    validate_team_policy_scripts(scripts)

    invalid = scripts[0].code.replace(
        "    api.stop_all()",
        "    api.carry_forward(distance_m=0.2, speed_mps=0.2, ramp_s=1.5)\n"
        "    api.stop_all()",
    )
    with pytest.raises(ValueError, match="exactly two expression statements"):
        validate_team_policy_scripts((replace(scripts[0], code=invalid), scripts[1]))


def test_linear_transport_compiles_one_shared_motion_window():
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "sync",
            "phase_id": "straight_transport",
            "synchronization_group": "straight_transport",
            "actions": {
                robot_id: {
                    "method": "coordinated_linear_transport",
                    "parameters": {
                        "direction": "forward",
                        "distance_m": 1.0,
                        "speed_mps": 0.5,
                        "ramp_s": 1.5,
                    },
                }
                for robot_id in ("front", "back")
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=4.0,
        minimum_lateral_transport_window_s=4.0,
        transport_duration_gain=0.0,
        transport_duration_baseline_m=1.0,
        minimum_transport_ramp_s=1.5,
    )
    motion_groups = [
        group
        for group in groups
        if group["front"].motion_axis == "longitudinal"
        and group["front"].speed > 0.0
        and group["front"].speed_start_mps is None
    ]
    brake_groups = [
        group
        for group in groups
        if group["front"].phase_id.endswith(".brake")
    ]

    assert len(motion_groups) == 1
    assert len(brake_groups) == 1
    motion = motion_groups[0]
    assert motion["front"].t == pytest.approx(motion["back"].t)
    assert motion["front"].duration_s == pytest.approx(motion["back"].duration_s)
    assert motion["front"].movement == motion["back"].movement == (1.0, 0.0, 0.0)
    brake = brake_groups[0]
    assert brake["front"].movement == brake["back"].movement == (0.0, 0.0, 0.0)
    assert brake["front"].speed == pytest.approx(-1.0)
    assert brake["front"].duration_s == pytest.approx(1.5)
    assert brake["front"].mode.name == "IDLE"


def test_short_linear_transport_uses_distance_speed_duration_without_hidden_window():
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "sync",
            "phase_id": "short_transport",
            "synchronization_group": "short_transport",
            "actions": {
                robot_id: {
                    "method": "coordinated_linear_transport",
                    "parameters": {
                        "direction": "forward",
                        "distance_m": 0.5,
                        "speed_mps": 0.5,
                        "ramp_s": 1.5,
                    },
                }
                for robot_id in ("front", "back")
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=0.4,
        transport_duration_gain=1.0,
        transport_duration_baseline_m=2.0,
        minimum_transport_ramp_s=1.5,
    )
    motion = next(
        group
        for group in groups
        if group["front"].motion_axis == "longitudinal"
        and group["front"].speed > 0.0
    )

    assert motion["front"].duration_s == pytest.approx(1.0)
    assert motion["back"].duration_s == pytest.approx(1.0)
    assert schedule_duration(groups) == pytest.approx(4.0)


def test_linear_transport_rejects_hidden_front_back_speed_compensation():
    payload = {
        "coordination_mode": "sync",
        "phase_id": "straight_transport",
        "synchronization_group": "straight_transport",
        "actions": {
            robot_id: {
                "method": "coordinated_linear_transport",
                "parameters": {
                    "direction": "forward",
                    "distance_m": 2.0,
                    "speed_mps": speed,
                    "ramp_s": 1.5,
                },
            }
            for robot_id, speed in (("front", 0.45), ("back", 0.50))
        },
    }

    with pytest.raises(ValueError, match="matching front/back speed_mps"):
        CoordinatedTeamPlan.from_payload(payload)


def test_backward_transport_uses_short_bounded_window_then_idle():
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "sync",
            "phase_id": "loaded_reverse",
            "synchronization_group": "loaded_reverse",
            "actions": {
                robot_id: {
                    "method": "coordinated_linear_transport",
                    "parameters": {
                        "direction": "backward",
                        "distance_m": 0.10,
                        "speed_mps": 0.20,
                        "ramp_s": 1.0,
                    },
                }
                for robot_id in ("front", "back")
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=1.0,
        transport_duration_gain=1.0,
        transport_duration_baseline_m=2.0,
        minimum_transport_ramp_s=1.0,
    )

    assert not any(
        command.speed_start_mps is not None
        for group in groups
        for command in group.values()
    )
    move_index = next(
        index
        for index, group in enumerate(groups)
        if group["front"].phase_id.endswith(".move")
    )
    move = groups[move_index]
    settle = groups[move_index + 1]
    assert move["front"].duration_s == pytest.approx(0.5)
    assert settle["front"].mode.name == "IDLE"
    assert settle["back"].mode.name == "IDLE"
    assert settle["front"].t == pytest.approx(
        move["front"].t + move["front"].duration_s
    )


def test_model_loaded_linear_allows_sync_backward_but_arc_rejects_it():
    validate_coordination_primitive_parameters(
        "coordinated_linear_transport",
        {
            "direction": "backward",
            "distance_m": 0.5,
            "speed_mps": 0.5,
            "ramp_s": 1.5,
        },
    )

    with pytest.raises(ValueError, match="travel_direction must be one of"):
        validate_coordination_primitive_parameters(
            "coordinated_arc_transport",
            {
                "turn_direction": "counterclockwise",
                "travel_direction": "backward",
                "angle_deg": 7.5,
                "duration_s": 4.0,
                "radius_m": 0.8,
            },
        )


def test_legacy_loaded_backward_rejects_async_team_dispatch():
    parameters = {
        "distance_m": 0.5,
        "speed_mps": 0.5,
        "ramp_s": 1.5,
    }
    payload = {
        "coordination_mode": "async",
        "phase_id": "loaded_reverse",
        "synchronization_group": "loaded_reverse",
        "actions": {
            "front": {
                "method": "carry_backward",
                "parameters": parameters,
            },
            "back": {
                "method": "hold_position",
                "parameters": {"hold_s": 2.0},
            },
        },
    }

    with pytest.raises(
        ValueError,
        match="loaded backward retreat requires both robots",
    ):
        CoordinatedTeamPlan.from_payload(payload)


def test_pivot_compiles_simultaneous_inner_rotation_and_outer_swing():
    shared = {
        "turn_direction": "clockwise",
        "travel_direction": "forward",
        "angle_deg": 10.0,
        "duration_s": 4.0,
        "swing_radius_m": 0.7,
    }
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "async",
            "phase_id": "payload_pivot",
            "synchronization_group": "payload_pivot",
            "actions": {
                "front": {
                    "method": "coordinated_pivot",
                    "parameters": {**shared, "role": "outer"},
                },
                "back": {
                    "method": "coordinated_pivot",
                    "parameters": {**shared, "role": "inner"},
                },
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=2.0,
        transport_duration_gain=0.0,
        transport_duration_baseline_m=1.0,
        minimum_transport_ramp_s=1.5,
    )
    pivot_groups = [group for group in groups if group["front"].motion_axis == "pivot"]

    assert pivot_groups
    for group in pivot_groups:
        outer = group["front"]
        inner = group["back"]
        assert outer.duration_s == pytest.approx(inner.duration_s)
        assert outer.delta_yaw_rad == pytest.approx(inner.delta_yaw_rad)
        assert outer.delta_yaw_rad < 0.0
        assert outer.speed > SONIC_ARC_TRANSLATION_ONSET_MPS
        assert outer.movement == (1.0, 0.0, 0.0)
        assert inner.speed == pytest.approx(0.0)
        assert inner.movement == (0.0, 0.0, 0.0)


def test_pivot_rejects_any_backward_outer_travel():
    shared = {
        "turn_direction": "counterclockwise",
        "travel_direction": "backward",
        "angle_deg": 7.5,
        "duration_s": 4.0,
        "swing_radius_m": 0.8,
    }
    with pytest.raises(ValueError, match="requires travel_direction='forward'"):
        CoordinatedTeamPlan.from_payload(
            {
                "coordination_mode": "async",
                "phase_id": "invalid_reverse_pivot",
                "synchronization_group": "invalid_reverse_pivot",
                "actions": {
                    "front": {
                        "method": "coordinated_pivot",
                        "parameters": {**shared, "role": "outer"},
                    },
                    "back": {
                        "method": "coordinated_pivot",
                        "parameters": {**shared, "role": "inner"},
                    },
                },
            }
        )


def test_alignment_compiles_relative_motion_without_sequential_calls():
    plan = CoordinatedTeamPlan.from_payload(
        {
            "coordination_mode": "async",
            "phase_id": "payload_alignment",
            "synchronization_group": "payload_alignment",
            "actions": {
                "front": {
                    "method": "coordinated_alignment",
                    "parameters": {
                        "axis": "longitudinal",
                        "delta": 0.20,
                        "duration_s": 3.0,
                    },
                },
                "back": {
                    "method": "coordinated_alignment",
                    "parameters": {
                        "axis": "longitudinal",
                        "delta": 0.0,
                        "duration_s": 3.0,
                    },
                },
            },
        }
    )

    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=1.0,
        minimum_lateral_transport_window_s=2.0,
        transport_duration_gain=0.0,
        transport_duration_baseline_m=1.0,
        minimum_transport_ramp_s=1.5,
    )

    assert groups
    assert all(set(group) == {"front", "back"} for group in groups)
    assert any(
        group["front"].movement == (1.0, 0.0, 0.0)
        and group["front"].speed
        == pytest.approx(calibrated_alignment_translation_speed(0.20, 3.0))
        and group["front"].speed >= SONIC_ARC_TRANSLATION_ONSET_MPS
        and group["back"].movement == (0.0, 0.0, 0.0)
        for group in groups
    )
    assert calibrated_alignment_translation_speed(0.15, 1.0) == pytest.approx(
        SONIC_ALIGNMENT_TRANSLATION_COMMAND_MPS
    )


def test_relative_alignment_feedback_stops_from_measured_catchup(tmp_path):
    state_file = tmp_path / "runtime_state.json"

    def write_state(*, front_x: float, back_x: float) -> None:
        state_file.write_text(
            json.dumps(
                {
                    "time_unix": time.time(),
                    "robots": {
                        "front": {"x": front_x, "y": 0.0, "yaw_rad": 0.0},
                        "back": {"x": back_x, "y": 1.0, "yaw_rad": 0.0},
                    },
                }
            ),
            encoding="utf-8",
        )

    write_state(front_x=0.0, back_x=0.0)
    feedback = RelativeAlignmentFeedback(
        state_file,
        mover_robot="back",
        reference_robot="front",
        axis="longitudinal",
        signed_target_delta_m=0.15,
        stop_margin_m=0.02,
    )

    write_state(front_x=0.01, back_x=0.13)
    assert feedback.reached() is False
    write_state(front_x=0.01, back_x=0.15)
    assert feedback.reached() is True
    report = feedback.report()
    assert report["max_progress_m"] == pytest.approx(0.14)
    assert report["early_completion_count"] == 1


def test_relative_alignment_feedback_uses_signed_lateral_body_axis(tmp_path):
    state_file = tmp_path / "runtime_state.json"

    def write_state(back_y: float) -> None:
        state_file.write_text(
            json.dumps(
                {
                    "time_unix": time.time(),
                    "robots": {
                        "front": {"x": 0.0, "y": 0.0, "yaw_rad": 0.0},
                        "back": {"x": 0.0, "y": back_y, "yaw_rad": 0.0},
                    },
                }
            ),
            encoding="utf-8",
        )

    write_state(1.0)
    feedback = RelativeAlignmentFeedback(
        state_file,
        mover_robot="back",
        reference_robot="front",
        axis="lateral",
        signed_target_delta_m=-0.10,
        stop_margin_m=0.01,
    )
    write_state(0.905)

    assert feedback.reached() is True
    assert feedback.report()["max_progress_m"] == pytest.approx(0.095)


def test_alignment_rejects_subobservable_yaw_delta():
    with pytest.raises(ValueError, match=r"0 or >= 2deg"):
        validate_coordination_primitive_parameters(
            "coordinated_alignment",
            {"axis": "yaw", "delta": 0.05, "duration_s": 2.0},
        )

    validate_coordination_primitive_parameters(
        "coordinated_alignment",
        {"axis": "yaw", "delta": 2.0, "duration_s": 2.0},
    )


def test_alignment_model_contract_rejects_loaded_body_backward_translation():
    with pytest.raises(ValueError, match="body-backward translation is forbidden"):
        validate_coordination_primitive_parameters(
            "coordinated_alignment",
            {"axis": "longitudinal", "delta": -0.15, "duration_s": 2.0},
        )

    for axis in ("lateral", "yaw"):
        validate_coordination_primitive_parameters(
            "coordinated_alignment",
            {
                "axis": axis,
                "delta": -0.15 if axis == "lateral" else -5.0,
                "duration_s": 2.0,
            },
        )


def test_alignment_team_plan_rejects_loaded_body_backward_translation():
    with pytest.raises(ValueError, match="body-backward translation is forbidden"):
        CoordinatedTeamPlan.from_payload(
            {
                "coordination_mode": "async",
                "phase_id": "invalid_backward_alignment",
                "synchronization_group": "invalid_backward_alignment",
                "actions": {
                    "front": {
                        "method": "coordinated_alignment",
                        "parameters": {
                            "axis": "longitudinal",
                            "delta": 0.15,
                            "duration_s": 2.0,
                        },
                    },
                    "back": {
                        "method": "coordinated_alignment",
                        "parameters": {
                            "axis": "longitudinal",
                            "delta": -0.15,
                            "duration_s": 2.0,
                        },
                    },
                },
            }
        )


def test_alignment_team_plan_rejects_divergent_loaded_lateral_motion():
    def payload(front_delta: float, back_delta: float) -> dict[str, object]:
        return {
            "coordination_mode": "async",
            "phase_id": "lateral_alignment",
            "synchronization_group": "lateral_alignment",
            "actions": {
                robot_id: {
                    "method": "coordinated_alignment",
                    "parameters": {
                        "axis": "lateral",
                        "delta": delta,
                        "duration_s": 2.0,
                    },
                }
                for robot_id, delta in {
                    "front": front_delta,
                    "back": back_delta,
                }.items()
            },
        }

    with pytest.raises(ValueError, match="cannot command divergent front/back motion"):
        CoordinatedTeamPlan.from_payload(payload(-0.15, 0.15))

    CoordinatedTeamPlan.from_payload(payload(-0.15, 0.0))


def test_arc_roles_enter_strategy_identity_and_dynamic_memory(tmp_path):
    scripts = _arc_scripts()
    policy = PolicyScript(
        script_id="box_carry_attempt_06",
        attempt=6,
        code=scripts[0].code,
        robot_scripts=scripts,
    )

    assert policy_strategy_signature(policy) == "async:coordinated_arc_transport"
    role_signature = json.loads(policy_role_signature(policy))
    assert {item["robot_id"]: item["profile_role"] for item in role_signature} == {
        "front": "outer",
        "back": "inner",
    }

    store = StrategyMemoryStore(tmp_path / "active_strategy.json")
    assert store.update_from_rollout(
        {
            "kind": "physical_rollout_finalized",
            "dry_run": False,
            "physically_dispatched": True,
            "physical_transaction_completed": True,
            "script_id": policy.script_id,
            "attempt": policy.attempt,
            "coordination_mode": "async",
            "executed_skill": "coordinated_arc_transport",
            "policy_execution_parameters": [
                {
                    "robot_id": "front",
                    "skill": "coordinated_arc_transport",
                    "parameters": _arc_parameters(0.8),
                },
                {
                    "robot_id": "back",
                    "skill": "coordinated_arc_transport",
                    "parameters": _arc_parameters(0.0),
                },
            ],
            "strategy_context": {"phase": "corridor_entry"},
            "verifier": {
                "step_accepted": True,
                "failure_type": "none",
                "transition_outcome": "ACCEPT_PROGRESS",
                "positive_route_memory_allowed": True,
                "checkpoint_before": "accepted_step_03",
                "checkpoint_after": "accepted_step_04",
            },
        }
    )
    learned = store.specs[0]
    assert learned["observed_coordination_primitive"] == "coordinated_arc_transport"
    assert "coordinated_arc_transport" in learned["decision"]
    assert "radius_m" not in json.dumps(learned, sort_keys=True)
