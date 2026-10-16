"""Build atomically published sync/async SONIC team command schedules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Any

from .commands import SonicCommand, SonicCommandStream, SonicLocomotionMode
from .coordination_primitives import (
    COORDINATION_PRIMITIVE_METHODS,
    COORDINATION_TRANSLATION_METHODS,
    build_coordination_primitive_stream,
    validate_coordination_primitive_team_actions,
)
from .skill_presets import SonicStaticSkillBuildConfig, build_static_sonic_skill_stream
from .upper_body import DEFAULT_CARRY_READY_POSE_NAME


COORDINATED_ACTION_METHODS = frozenset(
    {
        "hold_position",
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
        "turn_clockwise",
        "turn_counterclockwise",
    }
) | COORDINATION_PRIMITIVE_METHODS
ASYNC_ACTION_METHODS = COORDINATED_ACTION_METHODS - {"carry_backward"}
SYNC_PARAMETER_DELTA_LIMITS = {
    "distance_m": 0.05,
    "speed_mps": 0.05,
    "ramp_s": 0.50,
    "yaw_deg": 15.0,
    "yaw_rate_degps": 5.0,
    "turn_segment_deg": 3.0,
    "hold_s": 1.0,
}
TRANSLATION_METHODS = frozenset(
    {"carry_forward", "carry_backward", "carry_lateral_left", "carry_lateral_right"}
) | COORDINATION_TRANSLATION_METHODS


@dataclass(frozen=True)
class TeamAction:
    method: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class CoordinatedTeamPlan:
    coordination_mode: str
    phase_id: str
    synchronization_group: str
    actions: dict[str, TeamAction]

    @classmethod
    def from_payload(cls, payload: object) -> "CoordinatedTeamPlan":
        if not isinstance(payload, dict):
            raise ValueError("coordinated team plan must be a JSON object")
        mode = str(payload.get("coordination_mode", "")).strip().lower()
        if mode not in {"sync", "async"}:
            raise ValueError("coordinated team plan requires coordination_mode='sync' or 'async'")
        phase_id = str(payload.get("phase_id", "")).strip()
        sync_group = str(payload.get("synchronization_group", "")).strip()
        if not phase_id or not sync_group:
            raise ValueError("coordinated team plan requires phase_id and synchronization_group")
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, dict) or set(raw_actions) != {"front", "back"}:
            raise ValueError("coordinated team plan requires exactly front/back actions")
        actions: dict[str, TeamAction] = {}
        for robot_id in ("front", "back"):
            raw = raw_actions[robot_id]
            if not isinstance(raw, dict):
                raise ValueError(f"action for {robot_id} must be an object")
            method = str(raw.get("method", "")).strip()
            parameters = raw.get("parameters", {})
            if method not in COORDINATED_ACTION_METHODS:
                raise ValueError(
                    f"unsupported coordinated action api.{method}; "
                    f"allowed={sorted(COORDINATED_ACTION_METHODS)}"
                )
            if not isinstance(parameters, dict):
                raise ValueError(f"parameters for {robot_id} must be an object")
            actions[robot_id] = TeamAction(method=method, parameters=dict(parameters))
        validate_coordination_primitive_team_actions(
            actions,
            coordination_mode=mode,
        )
        if mode == "sync":
            validate_synchronized_actions(actions)
        else:
            validate_asynchronous_actions(actions)
        return cls(
            coordination_mode=mode,
            phase_id=phase_id,
            synchronization_group=sync_group,
            actions=actions,
        )


def validate_synchronized_actions(actions: dict[str, TeamAction]) -> None:
    """Allow small compensation differences inside one shared phase objective."""

    if set(actions) != {"front", "back"}:
        raise ValueError("sync coordination requires exactly front and back actions")
    methods = {item.method for item in actions.values()}
    if len(methods) != 1:
        raise ValueError(
            "sync coordination requires the same mover method and phase objective; "
            f"got {sorted(methods)}"
        )
    front = actions["front"].parameters
    back = actions["back"].parameters
    if set(front) != set(back):
        raise ValueError(
            "sync parameter compensation requires matching parameter names; "
            f"front={sorted(front)}, back={sorted(back)}"
        )
    violations: list[str] = []
    for name in sorted(front):
        front_value = front[name]
        back_value = back[name]
        if front_value == back_value:
            continue
        limit = SYNC_PARAMETER_DELTA_LIMITS.get(name)
        if limit is None:
            violations.append(f"{name} must match")
            continue
        try:
            delta = abs(float(front_value) - float(back_value))
        except (TypeError, ValueError):
            violations.append(f"{name} must be numeric when it differs")
            continue
        if delta > limit + 1.0e-9:
            violations.append(f"{name} delta {delta:g} exceeds {limit:g}")
    if violations:
        raise ValueError(
            "sync coordination permits only bounded front/back compensation: "
            + "; ".join(violations)
        )


def validate_asynchronous_actions(actions: dict[str, TeamAction]) -> None:
    """Require an executable role difference for an atomic async phase.

    The publish barrier is shared by both modes.  ``async`` is reserved for a
    different front/back action relationship; relabeling two identical mover
    calls must not disable the live formation controller used by ``sync``.
    """

    if set(actions) != {"front", "back"}:
        raise ValueError("async coordination requires exactly front and back actions")
    methods = {item.method for item in actions.values()}
    unsupported = methods - ASYNC_ACTION_METHODS
    if unsupported:
        raise ValueError(
            "async coordination contains unsupported actions: "
            + ", ".join(sorted(unsupported))
            + "; loaded backward retreat requires both robots in one sync transaction"
        )
    holders = [
        robot_id
        for robot_id, action in actions.items()
        if action.method == "hold_position"
    ]
    if len(holders) == 2:
        raise ValueError(
            "async coordination requires different executable roles; two hold_position "
            "calls form a sync hold"
        )
    if len(holders) == 1:
        return

    if len(methods) != 1:
        raise ValueError(
            "async coordination supports one mover skill plus an optional support hold; "
            f"got {sorted(methods)}"
        )
    if actions["front"].parameters == actions["back"].parameters:
        method = actions["front"].method
        raise ValueError(
            "async coordination requires different executable roles or parameters; "
            f"identical front/back api.{method} calls are sync"
        )
    try:
        validate_synchronized_actions(actions)
    except ValueError:
        return
    method = actions["front"].method
    raise ValueError(
        "async coordination requires a role-specific relation outside the bounded sync "
        f"compensation envelope; front/back api.{method} calls within that envelope are sync"
    )


def build_coordinated_schedule(
    plan: CoordinatedTeamPlan,
    *,
    minimum_transport_window_s: float,
    minimum_lateral_transport_window_s: float,
    transport_duration_gain: float,
    transport_duration_baseline_m: float,
    minimum_transport_ramp_s: float,
) -> list[dict[str, SonicCommand]]:
    """Compile per-robot actions into one complete command batch per interval."""

    streams = {
        robot_id: build_robot_action_stream(
            robot_id,
            action,
            sync_group=plan.synchronization_group,
            minimum_transport_window_s=minimum_transport_window_s,
            minimum_lateral_transport_window_s=minimum_lateral_transport_window_s,
            transport_duration_gain=transport_duration_gain,
            transport_duration_baseline_m=transport_duration_baseline_m,
            minimum_transport_ramp_s=minimum_transport_ramp_s,
        )
        for robot_id, action in plan.actions.items()
    }
    if (
        plan.coordination_mode == "sync"
        and all(action.method in TRANSLATION_METHODS for action in plan.actions.values())
    ):
        streams = _align_synchronized_translation_streams(streams)
    boundaries = {0.0}
    for stream in streams.values():
        for command in stream.commands:
            boundaries.add(float(command.t))
            boundaries.add(float(command.t + command.duration_s))
    ordered = sorted(boundaries)
    groups: list[dict[str, SonicCommand]] = []
    for start, end in zip(ordered, ordered[1:]):
        duration = end - start
        if duration <= 1.0e-9:
            continue
        sample_t = start + 0.5 * duration
        group: dict[str, SonicCommand] = {}
        for robot_id in ("front", "back"):
            active = _active_command(streams[robot_id], sample_t)
            if active is None:
                active = _idle_command(
                    robot_id=robot_id,
                    duration_s=duration,
                    phase_id=f"{plan.phase_id}.{robot_id}.support_hold",
                    sync_group=plan.synchronization_group,
                )
            else:
                original_start = float(active.t)
                continued_yaw_slice = (
                    active.delta_yaw_rad is not None and start > original_start + 1.0e-9
                )
                active = replace(
                    active,
                    t=start,
                    duration_s=duration,
                    phase_id=f"{plan.phase_id}.{robot_id}.{active.phase_id}",
                    phase=f"{plan.phase_id}.{robot_id}.{active.phase_id}",
                    sync_group=plan.synchronization_group,
                    delta_yaw_rad=(
                        0.0 if continued_yaw_slice else active.delta_yaw_rad
                    ),
                )
            group[robot_id] = active
        groups.append(group)
    return groups


def _align_synchronized_translation_streams(
    streams: dict[str, SonicCommandStream],
) -> dict[str, SonicCommandStream]:
    """Keep both loaded robots in the shared move window until it ends.

    Bounded front/back distance compensation may produce slightly different
    distance-derived move durations.  Letting the shorter stream enter its
    brake/hold phase while its partner is still translating removes that robot
    from formation feedback precisely when catch-up is needed.  A synchronous
    translation therefore uses the latest engage boundary, longest move
    duration, and longest brake duration for both streams.  Per-robot speed
    calibration remains intact and can still be adjusted by the live formation
    controller.
    """

    motion_commands: dict[str, SonicCommand] = {}
    for robot_id in ("front", "back"):
        stream = streams[robot_id]
        candidates = [
            command
            for command in stream.commands
            if _is_primary_translation_motion(command)
        ]
        if len(candidates) != 1:
            raise ValueError(
                "sync translation requires exactly one move command per robot; "
                f"robot={robot_id} count={len(candidates)}"
            )
        motion_commands[robot_id] = candidates[0]

    shared_move_start = max(command.t for command in motion_commands.values())
    shared_move_duration = max(command.duration_s for command in motion_commands.values())
    shared_move_end = shared_move_start + shared_move_duration
    soft_brakes = {
        robot_id: next(
            (
                command
                for command in streams[robot_id].commands
                if command.speed_start_mps is not None
                and command.speed_end_mps is not None
            ),
            None,
        )
        for robot_id in ("front", "back")
    }
    shared_soft_brake_duration = max(
        (
            command.duration_s
            for command in soft_brakes.values()
            if command is not None
        ),
        default=0.0,
    )
    shared_soft_brake_end = shared_move_end + shared_soft_brake_duration
    settle_commands = {
        robot_id: next(
            (
                command
                for command in streams[robot_id].commands
                if command.mode in {SonicLocomotionMode.IDLE, SonicLocomotionMode.IDLE_SQUAT}
                and command.t >= motion_commands[robot_id].t
                + motion_commands[robot_id].duration_s
                - 1.0e-9
            ),
            None,
        )
        for robot_id in ("front", "back")
    }
    shared_settle_duration = max(
        (
            command.duration_s
            for command in settle_commands.values()
            if command is not None
        ),
        default=0.0,
    )

    aligned: dict[str, SonicCommandStream] = {}
    for robot_id in ("front", "back"):
        stream = streams[robot_id]
        motion = motion_commands[robot_id]
        before = [
            command
            for command in stream.commands
            if command.t + command.duration_s <= motion.t + 1.0e-9
        ]
        commands: list[SonicCommand] = []
        if shared_move_start > 1.0e-9:
            engage = before[-1] if before else _stationary_translation_slice(
                motion,
                phase_suffix="engage",
            )
            commands.append(replace(engage, t=0.0, duration_s=shared_move_start))
        commands.append(
            replace(
                motion,
                t=shared_move_start,
                duration_s=shared_move_duration,
            )
        )
        soft_brake = soft_brakes[robot_id]
        if shared_soft_brake_duration > 1.0e-9 and soft_brake is not None:
            commands.append(
                replace(
                    soft_brake,
                    t=shared_move_end,
                    duration_s=shared_soft_brake_duration,
                )
            )
        settle = settle_commands[robot_id]
        if shared_settle_duration > 1.0e-9 and settle is not None:
            commands.append(
                replace(
                    settle,
                    t=shared_soft_brake_end,
                    duration_s=shared_settle_duration,
                )
            )
        aligned[robot_id] = SonicCommandStream(commands)
    return aligned


def _is_translation_motion(command: SonicCommand) -> bool:
    return (
        command.motion_axis in {"longitudinal", "lateral"}
        and command.mode not in {SonicLocomotionMode.IDLE, SonicLocomotionMode.IDLE_SQUAT}
        and command.speed > 0.0
        and hypot(command.movement[0], command.movement[1]) > 1.0e-9
    )


def _is_primary_translation_motion(command: SonicCommand) -> bool:
    return (
        _is_translation_motion(command)
        and command.speed_start_mps is None
        and command.speed_end_mps is None
    )


def _stationary_translation_slice(
    command: SonicCommand,
    *,
    phase_suffix: str,
) -> SonicCommand:
    base_phase = command.phase_id or command.phase or command.skill_node
    if base_phase.endswith(".move"):
        base_phase = base_phase[:-5]
    phase_id = f"{base_phase}.{phase_suffix}"
    return replace(
        command,
        mode=SonicLocomotionMode.IDLE,
        movement=(0.0, 0.0, 0.0),
        speed=-1.0,
        phase=phase_id,
        phase_id=phase_id,
    )


def build_robot_action_stream(
    robot_id: str,
    action: TeamAction,
    *,
    sync_group: str,
    minimum_transport_window_s: float,
    minimum_lateral_transport_window_s: float,
    transport_duration_gain: float,
    transport_duration_baseline_m: float,
    minimum_transport_ramp_s: float,
) -> SonicCommandStream:
    parameters = action.parameters
    if action.method == "hold_position":
        hold_s = max(0.5, min(10.0, float(parameters.get("hold_s", 2.0))))
        return SonicCommandStream(
            [
                _idle_command(
                    robot_id=robot_id,
                    duration_s=hold_s,
                    phase_id="hold_position",
                    sync_group=sync_group,
                )
            ]
        )

    if action.method in COORDINATION_PRIMITIVE_METHODS:
        return build_coordination_primitive_stream(
            robot_id,
            action.method,
            parameters,
            sync_group=sync_group,
            minimum_transport_window_s=minimum_transport_window_s,
            minimum_lateral_transport_window_s=minimum_lateral_transport_window_s,
            transport_duration_gain=transport_duration_gain,
            transport_duration_baseline_m=transport_duration_baseline_m,
            minimum_transport_ramp_s=minimum_transport_ramp_s,
        )

    config: dict[str, Any] = {
        "skill_name": action.method,
        "robot_ids": (robot_id,),
        "sync_group": sync_group,
    }
    if action.method in TRANSLATION_METHODS:
        distance = abs(float(parameters.get("distance_m", 0.05)))
        speed = abs(float(parameters.get("speed_mps", 0.20)))
        ramp = max(
            float(minimum_transport_ramp_s),
            float(parameters.get("ramp_s", minimum_transport_ramp_s)),
        )
        base_window = (
            minimum_lateral_transport_window_s
            if "lateral" in action.method
            else minimum_transport_window_s
        )
        extra_window = (
            max(0.0, distance - float(transport_duration_baseline_m))
            * max(0.0, float(transport_duration_gain))
            / max(speed, 1.0e-6)
        )
        config.update(
            {
                "distance_m": distance,
                "speed_mps": speed,
                "transport_ramp_seconds": ramp,
                "minimum_motion_duration_s": max(0.4, float(base_window) + extra_window),
            }
        )
    else:
        yaw = abs(float(parameters.get("yaw_deg", 5.0)))
        config.update(
            {
                "skill_name": "carry_turn_in_place",
                "yaw_deg": -yaw if action.method == "turn_clockwise" else yaw,
                "yaw_rate_degps": float(parameters.get("yaw_rate_degps", 5.0)),
                "turn_segment_deg": float(parameters.get("turn_segment_deg", 3.0)),
            }
        )
    return build_static_sonic_skill_stream(SonicStaticSkillBuildConfig(**config))


def _active_command(stream: SonicCommandStream, t: float) -> SonicCommand | None:
    for command in stream.commands:
        if command.t - 1.0e-9 <= t < command.t + command.duration_s - 1.0e-9:
            return command
    return None


def _idle_command(
    *,
    robot_id: str,
    duration_s: float,
    phase_id: str,
    sync_group: str,
) -> SonicCommand:
    return SonicCommand(
        t=0.0,
        robot_id=robot_id,
        skill_node="hold_position",
        mode=SonicLocomotionMode.IDLE,
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed=-1.0,
        height=0.80,
        duration_s=max(0.1, float(duration_s)),
        phase=phase_id,
        phase_id=phase_id,
        hand_pose_policy="runtime_upper_body_lock",
        hand_reference=DEFAULT_CARRY_READY_POSE_NAME,
        contact_intent="maintain_payload_support",
        sync_group=sync_group,
        motion_axis="hold",
        reference_frame="body",
    )


def schedule_duration(groups: list[dict[str, SonicCommand]]) -> float:
    return sum(
        max(command.duration_s for command in group.values())
        for group in groups
        if group
    )


def has_active_motion(group: dict[str, SonicCommand]) -> bool:
    return any(
        command.motion_axis
        in {"longitudinal", "lateral", "yaw", "arc", "pivot", "alignment_yaw"}
        and (
            command.delta_yaw_rad is not None
            or hypot(command.movement[0], command.movement[1]) > 1.0e-9
        )
        for command in group.values()
    )
