"""Compile rigid-payload coordination primitives into continuous SONIC streams."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, radians
from typing import Any

from .commands import SonicCommand, SonicCommandStream, SonicLocomotionMode
from .skill_presets import SonicStaticSkillBuildConfig, build_static_sonic_skill_stream
from .upper_body import DEFAULT_CARRY_READY_POSE_NAME


COORDINATED_LINEAR_TRANSPORT = "coordinated_linear_transport"
COORDINATED_ARC_TRANSPORT = "coordinated_arc_transport"
COORDINATED_PIVOT = "coordinated_pivot"
COORDINATED_ALIGNMENT = "coordinated_alignment"
MIN_COORDINATED_ARC_RADIUS_M = 0.05
SONIC_ARC_TRANSLATION_ONSET_MPS = 0.20
SONIC_MAX_TRANSLATION_COMMAND_MPS = 0.50
SONIC_ALIGNMENT_TRANSLATION_COMMAND_MPS = 0.20
ARC_TRAVEL_DIRECTIONS = frozenset({"forward", "backward", "left", "right"})

_BODY_FRAME_MOVEMENT_BY_DIRECTION = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
}

COORDINATION_PRIMITIVE_METHODS = frozenset(
    {
        COORDINATED_LINEAR_TRANSPORT,
        COORDINATED_ARC_TRANSPORT,
        COORDINATED_PIVOT,
        COORDINATED_ALIGNMENT,
    }
)
COORDINATION_TRANSLATION_METHODS = frozenset({COORDINATED_LINEAR_TRANSPORT})

_LINEAR_SKILL_BY_DIRECTION = {
    "forward": "carry_forward",
    "backward": "carry_backward",
    "left": "carry_lateral_left",
    "right": "carry_lateral_right",
}
_ARC_SETTLE_S = 1.0
_ARC_SEGMENT_DEG = 5.0
_BACKWARD_MINIMUM_MOTION_WINDOW_S = 0.4


def build_coordination_primitive_stream(
    robot_id: str,
    method: str,
    parameters: dict[str, Any],
    *,
    sync_group: str,
    minimum_transport_window_s: float,
    minimum_lateral_transport_window_s: float,
    transport_duration_gain: float,
    transport_duration_baseline_m: float,
    minimum_transport_ramp_s: float,
) -> SonicCommandStream:
    """Compile one robot's profile inside an atomically validated team primitive."""

    if method == COORDINATED_LINEAR_TRANSPORT:
        return _linear_stream(
            robot_id,
            parameters,
            sync_group=sync_group,
            minimum_transport_window_s=minimum_transport_window_s,
            minimum_lateral_transport_window_s=minimum_lateral_transport_window_s,
            transport_duration_gain=transport_duration_gain,
            transport_duration_baseline_m=transport_duration_baseline_m,
            minimum_transport_ramp_s=minimum_transport_ramp_s,
        )
    if method in {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}:
        return _arc_stream(
            robot_id,
            method,
            parameters,
            sync_group=sync_group,
        )
    if method == COORDINATED_ALIGNMENT:
        return _alignment_stream(
            robot_id,
            parameters,
            sync_group=sync_group,
        )
    raise ValueError(f"unsupported coordination primitive {method!r}")


def calibrated_arc_translation_speed(
    radius_m: float,
    angle_rad: float,
    duration_s: float,
) -> float:
    """Map geometric arc speed into SONIC's learned command domain."""

    radius = max(0.0, float(radius_m))
    if radius <= 1.0e-9:
        return 0.0
    geometric_speed = radius * abs(float(angle_rad)) / max(
        float(duration_s),
        1.0e-6,
    )
    return SONIC_ARC_TRANSLATION_ONSET_MPS + geometric_speed


def calibrated_alignment_translation_speed(
    delta_m: float,
    duration_s: float,
) -> float:
    """Use the lowest observable gait command while physics feedback bounds displacement."""

    geometric_speed = abs(float(delta_m)) / max(float(duration_s), 1.0e-6)
    return min(
        SONIC_MAX_TRANSLATION_COMMAND_MPS,
        max(SONIC_ALIGNMENT_TRANSLATION_COMMAND_MPS, geometric_speed),
    )


def validate_coordination_primitive_team_actions(
    actions: dict[str, object],
    *,
    coordination_mode: str,
) -> None:
    """Enforce cross-robot geometry before any planner command is published."""

    if set(actions) != {"front", "back"}:
        raise ValueError("coordination primitive requires exactly front and back actions")
    methods = {str(getattr(action, "method", "")) for action in actions.values()}
    primitive_methods = methods & COORDINATION_PRIMITIVE_METHODS
    if not primitive_methods:
        return
    if len(methods) != 1:
        raise ValueError(
            "one coordination primitive phase requires the same primitive family for both robots"
        )
    method = next(iter(methods))
    parameters = {
        robot_id: dict(getattr(action, "parameters", {}))
        for robot_id, action in actions.items()
    }

    if method == COORDINATED_LINEAR_TRANSPORT:
        if coordination_mode != "sync":
            raise ValueError("coordinated_linear_transport requires coordination_mode=sync")
        _require_shared(parameters, ("direction", "distance_m", "speed_mps", "ramp_s"))
        return

    if coordination_mode != "async":
        raise ValueError(f"{method} requires coordination_mode=async")

    if method == COORDINATED_ARC_TRANSPORT:
        _require_shared(
            parameters,
            ("turn_direction", "travel_direction", "angle_deg", "duration_s"),
        )
        radii = {
            robot_id: float(values.get("radius_m", -1.0))
            for robot_id, values in parameters.items()
        }
        if any(radius < MIN_COORDINATED_ARC_RADIUS_M for radius in radii.values()):
            raise ValueError(
                "coordinated_arc_transport requires radius_m >= "
                f"{MIN_COORDINATED_ARC_RADIUS_M:.2f} m for both robots; "
                "use coordinated_pivot for a zero-radius inner role"
            )
        if abs(radii["front"] - radii["back"]) < 0.05:
            raise ValueError(
                "coordinated_arc_transport requires distinct inner/outer radius_m values"
            )
        return

    if method == COORDINATED_PIVOT:
        _require_shared(
            parameters,
            (
                "turn_direction",
                "travel_direction",
                "angle_deg",
                "duration_s",
                "swing_radius_m",
            ),
        )
        roles = {robot_id: str(values.get("role", "")) for robot_id, values in parameters.items()}
        if set(roles.values()) != {"inner", "outer"}:
            raise ValueError(
                "coordinated_pivot requires exactly one role='inner' and one role='outer'"
            )
        travel_direction = str(parameters["front"].get("travel_direction", "forward"))
        if travel_direction != "forward":
            raise ValueError(
                "coordinated_pivot requires travel_direction='forward'; the inner robot "
                "must never be translated backward during a loaded corner turn"
            )
        return

    if method == COORDINATED_ALIGNMENT:
        _require_shared(parameters, ("axis", "duration_s"))
        axes = {str(values.get("axis", "")) for values in parameters.values()}
        deltas_by_robot = {
            robot_id: float(values.get("delta", 0.0))
            for robot_id, values in parameters.items()
        }
        if axes == {"longitudinal"}:
            backward_robots = sorted(
                robot_id
                for robot_id, delta in deltas_by_robot.items()
                if delta < 0.0
            )
            if backward_robots:
                raise ValueError(
                    "coordinated_alignment longitudinal delta must be >= 0 for both "
                    "robots; loaded body-backward translation is forbidden for "
                    f"{backward_robots}"
                )
        if axes == {"lateral"}:
            front_delta = deltas_by_robot["front"]
            back_delta = deltas_by_robot["back"]
            if front_delta * back_delta < 0.0:
                raise ValueError(
                    "loaded lateral alignment cannot command divergent front/back motion; "
                    "use one reference robot or coordinated_linear_transport for lateral travel"
                )
        deltas = list(deltas_by_robot.values())
        if abs(deltas[0] - deltas[1]) < 1.0e-6:
            raise ValueError(
                "coordinated_alignment requires a relative front/back delta, not identical calls"
            )


def _require_shared(
    parameters: dict[str, dict[str, Any]],
    names: tuple[str, ...],
) -> None:
    for name in names:
        values = {repr(payload.get(name)) for payload in parameters.values()}
        if len(values) != 1:
            raise ValueError(
                f"coordination primitive requires matching front/back {name}; got "
                f"{ {robot_id: payload.get(name) for robot_id, payload in parameters.items()} }"
            )


def _linear_stream(
    robot_id: str,
    parameters: dict[str, Any],
    *,
    sync_group: str,
    minimum_transport_window_s: float,
    minimum_lateral_transport_window_s: float,
    transport_duration_gain: float,
    transport_duration_baseline_m: float,
    minimum_transport_ramp_s: float,
) -> SonicCommandStream:
    direction = str(parameters["direction"])
    static_skill = _LINEAR_SKILL_BY_DIRECTION[direction]
    distance = abs(float(parameters["distance_m"]))
    speed = abs(float(parameters["speed_mps"]))
    ramp = max(float(minimum_transport_ramp_s), float(parameters["ramp_s"]))
    base_window = (
        minimum_lateral_transport_window_s
        if direction in {"left", "right"}
        else (
            min(minimum_transport_window_s, _BACKWARD_MINIMUM_MOTION_WINDOW_S)
            if direction == "backward"
            else minimum_transport_window_s
        )
    )
    extra_window = (
        max(0.0, distance - float(transport_duration_baseline_m))
        * max(0.0, float(transport_duration_gain))
        / max(speed, 1.0e-6)
    )
    static_stream = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name=static_skill,
            robot_ids=(robot_id,),
            distance_m=distance,
            speed_mps=speed,
            transport_ramp_seconds=ramp,
            minimum_motion_duration_s=max(0.4, float(base_window) + extra_window),
            sync_group=sync_group,
        )
    )
    # Preserve the manually verified command envelope exactly: one stationary
    # engage, one fixed-speed move, then one stationary brake. Speeds below the
    # learned gait onset are not a physical velocity ramp in SONIC and changing
    # speed every publisher sample forces repeated policy replanning.
    return _retag_stream(static_stream, COORDINATED_LINEAR_TRANSPORT)


def _arc_stream(
    robot_id: str,
    method: str,
    parameters: dict[str, Any],
    *,
    sync_group: str,
) -> SonicCommandStream:
    angle_deg = abs(float(parameters["angle_deg"]))
    signed_angle_rad = radians(angle_deg)
    if str(parameters["turn_direction"]) == "clockwise":
        signed_angle_rad = -signed_angle_rad
    duration_s = float(parameters["duration_s"])
    travel_direction = str(parameters.get("travel_direction", "forward"))
    if travel_direction not in ARC_TRAVEL_DIRECTIONS:
        raise ValueError(
            "coordinated arc travel_direction must be one of "
            f"{sorted(ARC_TRAVEL_DIRECTIONS)}, got {travel_direction!r}"
        )
    if method == COORDINATED_PIVOT and travel_direction != "forward":
        raise ValueError(
            "coordinated_pivot requires travel_direction='forward' for the outer role"
        )
    radius_m = (
        float(parameters["radius_m"])
        if method == COORDINATED_ARC_TRANSPORT
        else (
            0.0
            if str(parameters["role"]) == "inner"
            else float(parameters["swing_radius_m"])
        )
    )
    speed_mps = calibrated_arc_translation_speed(
        radius_m,
        signed_angle_rad,
        duration_s,
    )
    if speed_mps > SONIC_MAX_TRANSLATION_COMMAND_MPS + 1.0e-9:
        raise ValueError(
            f"calibrated arc command speed {speed_mps:.3f}m/s exceeds "
            f"{SONIC_MAX_TRANSLATION_COMMAND_MPS:.2f}m/s"
        )
    segment_count = max(1, int(ceil(angle_deg / _ARC_SEGMENT_DEG)))
    segment_duration_s = duration_s / segment_count
    segment_delta_yaw = signed_angle_rad / segment_count
    commands = [
        _coordination_command(
            t=0.0,
            robot_id=robot_id,
            method=method,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            speed=-1.0,
            duration_s=_ARC_SETTLE_S,
            phase_id=f"{method}.engage",
            sync_group=sync_group,
            motion_axis="hold",
        )
    ]
    for segment_index in range(1, segment_count + 1):
        commands.append(
            _coordination_command(
                t=_ARC_SETTLE_S + (segment_index - 1) * segment_duration_s,
                robot_id=robot_id,
                method=method,
                mode=SonicLocomotionMode.SLOW_WALK,
                movement=(
                    _BODY_FRAME_MOVEMENT_BY_DIRECTION[travel_direction]
                    if radius_m > 1.0e-6
                    else (0.0, 0.0, 0.0)
                ),
                speed=speed_mps,
                duration_s=segment_duration_s,
                phase_id=f"{method}.arc_{segment_index:02d}_of_{segment_count:02d}",
                sync_group=sync_group,
                motion_axis="arc" if method == COORDINATED_ARC_TRANSPORT else "pivot",
                delta_yaw_rad=segment_delta_yaw,
                target_yaw_rad=signed_angle_rad * segment_index / segment_count,
            )
        )
    commands.append(
        _coordination_command(
            t=_ARC_SETTLE_S + duration_s,
            robot_id=robot_id,
            method=method,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            speed=-1.0,
            duration_s=_ARC_SETTLE_S,
            phase_id=f"{method}.brake",
            sync_group=sync_group,
            motion_axis="hold",
        )
    )
    return SonicCommandStream(commands)


def _alignment_stream(
    robot_id: str,
    parameters: dict[str, Any],
    *,
    sync_group: str,
) -> SonicCommandStream:
    axis = str(parameters["axis"])
    delta = float(parameters["delta"])
    duration_s = float(parameters["duration_s"])
    if abs(delta) <= 1.0e-9:
        return SonicCommandStream(
            [
                _coordination_command(
                    t=0.0,
                    robot_id=robot_id,
                    method=COORDINATED_ALIGNMENT,
                    mode=SonicLocomotionMode.IDLE,
                    movement=(0.0, 0.0, 0.0),
                    speed=-1.0,
                    duration_s=duration_s,
                    phase_id="coordinated_alignment.reference_hold",
                    sync_group=sync_group,
                    motion_axis="hold",
                )
            ]
        )
    if axis == "yaw":
        signed_angle_rad = radians(delta)
        segment_count = max(1, int(ceil(abs(delta) / _ARC_SEGMENT_DEG)))
        segment_duration_s = duration_s / segment_count
        return SonicCommandStream(
            [
                _coordination_command(
                    t=(index - 1) * segment_duration_s,
                    robot_id=robot_id,
                    method=COORDINATED_ALIGNMENT,
                    mode=SonicLocomotionMode.SLOW_WALK,
                    movement=(0.0, 0.0, 0.0),
                    speed=0.0,
                    duration_s=segment_duration_s,
                    phase_id=f"coordinated_alignment.yaw_{index:02d}_of_{segment_count:02d}",
                    sync_group=sync_group,
                    motion_axis="alignment_yaw",
                    delta_yaw_rad=signed_angle_rad / segment_count,
                    target_yaw_rad=signed_angle_rad * index / segment_count,
                )
                for index in range(1, segment_count + 1)
            ]
        )

    direction = (
        "forward"
        if axis == "longitudinal" and delta > 0.0
        else "backward"
        if axis == "longitudinal"
        else "left"
        if delta > 0.0
        else "right"
    )
    stream = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name=_LINEAR_SKILL_BY_DIRECTION[direction],
            robot_ids=(robot_id,),
            distance_m=abs(delta),
            speed_mps=calibrated_alignment_translation_speed(delta, duration_s),
            minimum_motion_duration_s=duration_s,
            transport_ramp_seconds=0.0,
            sync_group=sync_group,
        )
    )
    return _retag_stream(stream, COORDINATED_ALIGNMENT)


def _retag_stream(stream: SonicCommandStream, method: str) -> SonicCommandStream:
    return SonicCommandStream(
        [
            replace(
                command,
                skill_node=method,
                phase=f"{method}.{command.phase}",
                phase_id=f"{method}.{command.phase_id}",
            )
            for command in stream.commands
        ]
    )


def _coordination_command(
    *,
    t: float,
    robot_id: str,
    method: str,
    mode: SonicLocomotionMode,
    movement: tuple[float, float, float],
    speed: float,
    duration_s: float,
    phase_id: str,
    sync_group: str,
    motion_axis: str,
    delta_yaw_rad: float | None = None,
    target_yaw_rad: float | None = None,
) -> SonicCommand:
    return SonicCommand(
        t=t,
        robot_id=robot_id,
        skill_node=method,
        mode=mode,
        movement=movement,
        facing=(1.0, 0.0, 0.0),
        speed=speed,
        height=0.80,
        duration_s=max(0.1, duration_s),
        ramp_s=(
            _ARC_SETTLE_S
            if method
            in {
                COORDINATED_ARC_TRANSPORT,
                COORDINATED_PIVOT,
            }
            else 0.0
        ),
        phase=phase_id,
        phase_id=phase_id,
        hand_pose_policy="runtime_upper_body_lock",
        hand_reference=DEFAULT_CARRY_READY_POSE_NAME,
        contact_intent="maintain_payload_support",
        sync_group=sync_group,
        abort_on_contact_loss=True,
        allowed_contact_loss_s=0.3,
        motion_axis=motion_axis,
        reference_frame="body",
        delta_yaw_rad=delta_yaw_rad,
        target_yaw_rad=target_yaw_rad,
    )


__all__ = [
    "ARC_TRAVEL_DIRECTIONS",
    "COORDINATED_ALIGNMENT",
    "COORDINATED_ARC_TRANSPORT",
    "COORDINATED_LINEAR_TRANSPORT",
    "COORDINATED_PIVOT",
    "COORDINATION_PRIMITIVE_METHODS",
    "COORDINATION_TRANSLATION_METHODS",
    "MIN_COORDINATED_ARC_RADIUS_M",
    "SONIC_ARC_TRANSLATION_ONSET_MPS",
    "SONIC_MAX_TRANSLATION_COMMAND_MPS",
    "build_coordination_primitive_stream",
    "calibrated_alignment_translation_speed",
    "calibrated_arc_translation_speed",
    "validate_coordination_primitive_team_actions",
]
