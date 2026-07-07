"""Static SONIC skill presets used before SkillGraph evolution.

These presets are deliberately narrow. They describe the stable deployment
skills we are validating with the official GEAR-SONIC process backend, while
leaving the high-frequency locomotion and balance to SONIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, pi, radians
from typing import Literal

from praxis_coop.sonic.commands import SonicCommand, SonicCommandStream, SonicLocomotionMode


SkillName = Literal[
    "parallel_forward",
    "carry_forward",
    "parallel_backward",
    "carry_backward",
    "parallel_lateral_left",
    "parallel_lateral_right",
    "carry_lateral_left",
    "carry_lateral_right",
    "turn_in_place",
    "carry_turn_in_place",
    "carry_place_crouch",
    "carry_recover_stand",
]


MODE_BY_NAME = {
    "idle": SonicLocomotionMode.IDLE,
    "idle_squat": SonicLocomotionMode.IDLE_SQUAT,
    "squat": SonicLocomotionMode.IDLE_SQUAT,
    "slow_walk": SonicLocomotionMode.SLOW_WALK,
    "walk": SonicLocomotionMode.WALK,
    "run": SonicLocomotionMode.RUN,
    "object_carrying": SonicLocomotionMode.OBJECT_CARRYING,
}


@dataclass(frozen=True)
class SonicStaticSkill:
    name: str
    description: str
    requires_upper_body_lock: bool
    default_mode: SonicLocomotionMode
    default_speed_mps: float
    default_distance_m: float = 0.0
    default_height: float = -1.0


@dataclass(frozen=True)
class SonicStaticSkillBuildConfig:
    skill_name: str
    robot_ids: tuple[str, ...] = ("front", "back")
    facing_xy: tuple[float, float] = (1.0, 0.0)
    movement_xy: tuple[float, float] | None = None
    distance_m: float | None = None
    speed_mps: float | None = None
    mode_name: str | None = None
    height: float | None = None
    yaw_deg: float = 90.0
    yaw_rate_degps: float = 8.0
    turn_segment_deg: float = 5.0
    turn_command_scale: float = 1.0
    place_start_height: float = 0.80
    place_height: float = 0.55
    place_ramp_seconds: float = 5.0
    place_hold_seconds: float = 4.0
    place_rise: bool = False
    sync_group: str = "static_sonic_skill"


STATIC_SONIC_SKILLS: dict[str, SonicStaticSkill] = {
    "parallel_forward": SonicStaticSkill(
        name="parallel_forward",
        description="Two G1s, or one selected G1, walk forward with no arm lock requirement.",
        requires_upper_body_lock=False,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.20,
        default_distance_m=2.0,
    ),
    "carry_forward": SonicStaticSkill(
        name="carry_forward",
        description="Move forward while the runtime keeps the tuned carry-ready arm pose locked.",
        requires_upper_body_lock=True,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.15,
        default_distance_m=1.5,
    ),
    "parallel_backward": SonicStaticSkill(
        name="parallel_backward",
        description="Back up with no arm lock requirement.",
        requires_upper_body_lock=False,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.12,
        default_distance_m=1.5,
    ),
    "carry_backward": SonicStaticSkill(
        name="carry_backward",
        description="Back up while preserving the runtime carry-ready arm pose.",
        requires_upper_body_lock=True,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.10,
        default_distance_m=1.5,
    ),
    "parallel_lateral_left": SonicStaticSkill(
        name="parallel_lateral_left",
        description="Move left relative to the current facing direction with no arm lock requirement.",
        requires_upper_body_lock=False,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.45,
        default_distance_m=1.5,
    ),
    "parallel_lateral_right": SonicStaticSkill(
        name="parallel_lateral_right",
        description="Move right relative to the current facing direction with no arm lock requirement.",
        requires_upper_body_lock=False,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.45,
        default_distance_m=1.5,
    ),
    "carry_lateral_left": SonicStaticSkill(
        name="carry_lateral_left",
        description="Move left relative to facing while preserving the tuned carry-ready arm pose.",
        requires_upper_body_lock=True,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.45,
        default_distance_m=1.5,
    ),
    "carry_lateral_right": SonicStaticSkill(
        name="carry_lateral_right",
        description="Move right relative to facing while preserving the tuned carry-ready arm pose.",
        requires_upper_body_lock=True,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.45,
        default_distance_m=1.5,
    ),
    "turn_in_place": SonicStaticSkill(
        name="turn_in_place",
        description="Turn the body in place by a requested yaw angle.",
        requires_upper_body_lock=False,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.0,
    ),
    "carry_turn_in_place": SonicStaticSkill(
        name="carry_turn_in_place",
        description="Turn in place by a requested yaw angle while preserving the carry-ready arm pose.",
        requires_upper_body_lock=True,
        default_mode=SonicLocomotionMode.SLOW_WALK,
        default_speed_mps=0.0,
    ),
    "carry_place_crouch": SonicStaticSkill(
        name="carry_place_crouch",
        description=(
            "Stationary knee-bend place skill. The deploy planner stays in IDLE; "
            "the MuJoCo runtime applies the lower-body place posture from the heartbeat."
        ),
        requires_upper_body_lock=True,
        default_mode=SonicLocomotionMode.IDLE,
        default_speed_mps=0.0,
        default_height=0.80,
    ),
    "carry_recover_stand": SonicStaticSkill(
        name="carry_recover_stand",
        description=(
            "Recover to upright standing after carry_place_crouch while preserving "
            "the tuned carry-ready arm pose."
        ),
        requires_upper_body_lock=True,
        default_mode=SonicLocomotionMode.IDLE,
        default_speed_mps=0.0,
        default_height=0.80,
    ),
}


def list_static_sonic_skills() -> list[SonicStaticSkill]:
    return [STATIC_SONIC_SKILLS[name] for name in sorted(STATIC_SONIC_SKILLS)]


def build_static_sonic_skill_stream(config: SonicStaticSkillBuildConfig) -> SonicCommandStream:
    """Build a timestamped command stream for one static skill.

    Arm and hand preservation is handled by the MuJoCo runtime's motor-side
    upper-body lock. These commands intentionally do not send planner-side upper
    body targets, so locomotion phases cannot overwrite the locked arm pose.
    """

    if config.skill_name not in STATIC_SONIC_SKILLS:
        choices = ", ".join(sorted(STATIC_SONIC_SKILLS))
        raise ValueError(f"unknown static SONIC skill {config.skill_name!r}; expected one of: {choices}")
    skill = STATIC_SONIC_SKILLS[config.skill_name]
    robot_ids = tuple(config.robot_ids)
    if not robot_ids:
        raise ValueError("at least one robot id is required")
    facing = _unit_xy(*config.facing_xy)
    mode = MODE_BY_NAME[config.mode_name] if config.mode_name else skill.default_mode
    height = skill.default_height if config.height is None else float(config.height)

    if config.skill_name in {"parallel_forward", "carry_forward"}:
        distance = skill.default_distance_m if config.distance_m is None else abs(float(config.distance_m))
        speed = skill.default_speed_mps if config.speed_mps is None else abs(float(config.speed_mps))
        commands = _translation_commands(
            robot_ids=robot_ids,
            skill=skill,
            mode=mode,
            movement=facing,
            facing=facing,
            distance_m=distance,
            speed_mps=speed,
            height=height,
            phase_id=f"{config.skill_name}.forward",
            motion_axis="longitudinal",
            reference_frame="body",
            sync_group=config.sync_group,
        )
        return SonicCommandStream(commands)

    if config.skill_name in {"parallel_backward", "carry_backward"}:
        distance = skill.default_distance_m if config.distance_m is None else abs(float(config.distance_m))
        speed = skill.default_speed_mps if config.speed_mps is None else abs(float(config.speed_mps))
        movement = (-facing[0], -facing[1])
        commands = _translation_commands(
            robot_ids=robot_ids,
            skill=skill,
            mode=mode,
            movement=movement,
            facing=facing,
            distance_m=distance,
            speed_mps=speed,
            height=height,
            phase_id=f"{config.skill_name}.backward",
            motion_axis="longitudinal",
            reference_frame="body",
            sync_group=config.sync_group,
        )
        return SonicCommandStream(commands)

    if config.skill_name in {
        "parallel_lateral_left",
        "parallel_lateral_right",
        "carry_lateral_left",
        "carry_lateral_right",
    }:
        distance = skill.default_distance_m if config.distance_m is None else abs(float(config.distance_m))
        speed = skill.default_speed_mps if config.speed_mps is None else abs(float(config.speed_mps))
        side_sign = 1.0 if config.skill_name.endswith("_left") else -1.0
        movement = _unit_xy(*config.movement_xy) if config.movement_xy is not None else _lateral_xy(facing, side_sign=side_sign)
        commands = _translation_commands(
            robot_ids=robot_ids,
            skill=skill,
            mode=mode,
            movement=movement,
            facing=facing,
            distance_m=distance,
            speed_mps=speed,
            height=height,
            phase_id=f"{config.skill_name}.lateral",
            motion_axis="lateral",
            reference_frame="body",
            sync_group=config.sync_group,
        )
        return SonicCommandStream(commands)

    if config.skill_name in {"turn_in_place", "carry_turn_in_place"}:
        return SonicCommandStream(
            _turn_commands(
                robot_ids=robot_ids,
                skill=skill,
                mode=mode,
                facing=facing,
                yaw_rad=radians(float(config.yaw_deg) * float(config.turn_command_scale)),
                yaw_rate_radps=radians(max(1.0e-6, abs(float(config.yaw_rate_degps)))),
                segment_yaw_rad=radians(max(1.0, abs(float(config.turn_segment_deg)))),
                height=height,
                sync_group=config.sync_group,
            )
        )

    if config.skill_name == "carry_place_crouch":
        return SonicCommandStream(
            _place_crouch_commands(
                robot_ids=robot_ids,
                skill=skill,
                facing=facing,
                start_height=float(config.place_start_height),
                place_height=float(config.place_height),
                ramp_seconds=max(0.1, float(config.place_ramp_seconds)),
                hold_seconds=max(0.0, float(config.place_hold_seconds)),
                rise=config.place_rise,
                sync_group=config.sync_group,
            )
        )

    if config.skill_name == "carry_recover_stand":
        return SonicCommandStream(
            _recover_stand_commands(
                robot_ids=robot_ids,
                skill=skill,
                facing=facing,
                stand_height=height,
                ramp_seconds=max(0.1, float(config.place_ramp_seconds)),
                hold_seconds=max(0.0, float(config.place_hold_seconds)),
                sync_group=config.sync_group,
            )
        )

    raise AssertionError(f"unhandled static SONIC skill: {config.skill_name}")


def skill_requires_upper_body_lock(skill_name: str) -> bool:
    return STATIC_SONIC_SKILLS[skill_name].requires_upper_body_lock


def _translation_commands(
    *,
    robot_ids: tuple[str, ...],
    skill: SonicStaticSkill,
    mode: SonicLocomotionMode,
    movement: tuple[float, float],
    facing: tuple[float, float],
    distance_m: float,
    speed_mps: float,
    height: float,
    phase_id: str,
    motion_axis: str,
    reference_frame: str,
    sync_group: str,
) -> list[SonicCommand]:
    duration_s = max(0.4, distance_m / max(speed_mps, 1.0e-6)) if distance_m > 1.0e-6 else 0.4
    movement_3d = (movement[0], movement[1], 0.0) if distance_m > 1.0e-6 else (0.0, 0.0, 0.0)
    return [
        _command(
            robot_id=robot_id,
            skill=skill,
            mode=mode,
            movement=movement_3d,
            facing=(facing[0], facing[1], 0.0),
            speed=speed_mps,
            height=height,
            duration_s=duration_s,
            phase_id=phase_id,
            sync_group=sync_group,
            motion_axis=motion_axis,
            reference_frame=reference_frame,
        )
        for robot_id in robot_ids
    ]


def _turn_commands(
    *,
    robot_ids: tuple[str, ...],
    skill: SonicStaticSkill,
    mode: SonicLocomotionMode,
    facing: tuple[float, float],
    yaw_rad: float,
    yaw_rate_radps: float,
    segment_yaw_rad: float,
    height: float,
    sync_group: str,
) -> list[SonicCommand]:
    segment_yaw = min(max(segment_yaw_rad, radians(1.0)), pi / 2.0)
    segment_count = max(1, int(ceil(abs(yaw_rad) / segment_yaw)))
    duration_s = max(0.4, abs(yaw_rad) / segment_count / max(yaw_rate_radps, 1.0e-6))
    segment_delta_yaw = yaw_rad / segment_count
    commands: list[SonicCommand] = []
    for segment_idx in range(1, segment_count + 1):
        t = (segment_idx - 1) * duration_s
        alpha = segment_idx / segment_count
        cumulative_delta_yaw = yaw_rad * alpha
        for robot_id in robot_ids:
            commands.append(
                _command(
                    t=t,
                    robot_id=robot_id,
                    skill=skill,
                    mode=mode,
                    movement=(0.0, 0.0, 0.0),
                    facing=(facing[0], facing[1], 0.0),
                    speed=0.0,
                    height=height,
                    duration_s=duration_s,
                    phase_id=f"{skill.name}.turn_{segment_idx:02d}_of_{segment_count:02d}",
                    sync_group=sync_group,
                    motion_axis="yaw",
                    delta_yaw_rad=segment_delta_yaw,
                    target_yaw_rad=cumulative_delta_yaw,
                )
            )
    return commands


def _place_crouch_commands(
    *,
    robot_ids: tuple[str, ...],
    skill: SonicStaticSkill,
    facing: tuple[float, float],
    start_height: float,
    place_height: float,
    ramp_seconds: float,
    hold_seconds: float,
    rise: bool,
    sync_group: str,
) -> list[SonicCommand]:
    heights = [place_height, place_height]
    phase_ids = [f"{skill.name}.lower", f"{skill.name}.hold"]
    durations = [ramp_seconds, hold_seconds]
    if rise:
        heights.append(start_height)
        phase_ids.append(f"{skill.name}.rise")
        durations.append(ramp_seconds)
    commands: list[SonicCommand] = []
    t = 0.0
    for height, phase_id, duration_s in zip(heights, phase_ids, durations):
        duration = max(0.1, duration_s)
        for robot_id in robot_ids:
            commands.append(
                _command(
                    t=t,
                    robot_id=robot_id,
                    skill=skill,
                    mode=SonicLocomotionMode.IDLE,
                    movement=(0.0, 0.0, 0.0),
                    facing=(facing[0], facing[1], 0.0),
                    speed=-1.0,
                    height=height,
                    duration_s=duration,
                    phase_id=phase_id,
                    sync_group=sync_group,
                    motion_axis="height",
                )
            )
        t += duration
    return commands


def _recover_stand_commands(
    *,
    robot_ids: tuple[str, ...],
    skill: SonicStaticSkill,
    facing: tuple[float, float],
    stand_height: float,
    ramp_seconds: float,
    hold_seconds: float,
    sync_group: str,
) -> list[SonicCommand]:
    phase_ids = [f"{skill.name}.recover", f"{skill.name}.hold"]
    durations = [ramp_seconds, hold_seconds]
    commands: list[SonicCommand] = []
    t = 0.0
    for phase_id, duration_s in zip(phase_ids, durations):
        duration = max(0.1, duration_s)
        for robot_id in robot_ids:
            commands.append(
                _command(
                    t=t,
                    robot_id=robot_id,
                    skill=skill,
                    mode=SonicLocomotionMode.IDLE,
                    movement=(0.0, 0.0, 0.0),
                    facing=(facing[0], facing[1], 0.0),
                    speed=-1.0,
                    height=stand_height,
                    duration_s=duration,
                    phase_id=phase_id,
                    sync_group=sync_group,
                    motion_axis="height",
                )
            )
        t += duration
    return commands


def _command(
    *,
    robot_id: str,
    skill: SonicStaticSkill,
    mode: SonicLocomotionMode,
    movement: tuple[float, float, float],
    facing: tuple[float, float, float],
    speed: float,
    height: float,
    duration_s: float,
    phase_id: str,
    sync_group: str,
    motion_axis: str,
    t: float = 0.0,
    delta_yaw_rad: float | None = None,
    target_yaw_rad: float | None = None,
    reference_frame: str = "world",
) -> SonicCommand:
    return SonicCommand(
        t=t,
        robot_id=robot_id,
        skill_node=skill.name,
        mode=mode,
        movement=movement,
        facing=facing,
        speed=speed,
        height=height,
        duration_s=duration_s,
        phase=phase_id,
        phase_id=phase_id,
        hand_pose_policy="runtime_upper_body_lock" if skill.requires_upper_body_lock else "none",
        hand_reference="elbow110_carry_ready" if skill.requires_upper_body_lock else "",
        sync_group=sync_group,
        motion_axis=motion_axis,
        reference_frame=reference_frame,
        delta_yaw_rad=delta_yaw_rad,
        target_yaw_rad=target_yaw_rad,
    )


def _unit_xy(x: float, y: float) -> tuple[float, float]:
    norm = hypot(x, y)
    if norm <= 1.0e-9:
        raise ValueError("facing vector cannot be zero")
    return (x / norm, y / norm)


def _lateral_xy(facing: tuple[float, float], *, side_sign: float) -> tuple[float, float]:
    return (-facing[1] * side_sign, facing[0] * side_sign)
