"""Upper-body and hand pose presets for GEAR-SONIC planner commands."""

from __future__ import annotations

from collections.abc import Iterable


# Order expected by GEAR-SONIC's ZMQ planner `upper_body_position` field.
# It matches `upper_body_joint_isaaclab_order_in_mujoco_index` in
# `gear_sonic_deploy/.../policy_parameters.hpp`.
UPPER_BODY_JOINT_NAMES = (
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
)

DEX3_HAND_JOINT_NAMES = (
    "thumb_0",
    "thumb_1",
    "index",
    "middle",
    "ring",
    "little",
    "wrist_aux",
)

ZERO_UPPER_BODY = (0.0,) * len(UPPER_BODY_JOINT_NAMES)

SONIC_DEFAULT_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.2,
    -0.2,
    0.0,
    0.0,
    0.6,
    0.6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

# Conservative initial target for a payload-support pose.  In the G1 MJCF/SONIC
# convention, negative shoulder pitch moves the wrists in front of the torso.
SOFT_HORIZONTAL_CARRY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.80,
    -0.80,
    0.16,
    -0.16,
    0.0,
    0.0,
    0.95,
    0.95,
    0.0,
    0.0,
    -0.20,
    -0.20,
    0.0,
    0.0,
)

HORIZONTAL_CARRY_UPPER_BODY = SOFT_HORIZONTAL_CARRY_UPPER_BODY

# First stable stepping stone for SONIC: keep the arms modestly forward while
# preserving enough shoulder freedom for balance recovery.
BALANCE_CARRY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.35,
    -0.35,
    0.10,
    -0.10,
    0.0,
    0.0,
    0.80,
    0.80,
    0.0,
    0.0,
    -0.08,
    -0.08,
    0.0,
    0.0,
)

UPPER_BODY_POSES = {
    "zero": ZERO_UPPER_BODY,
    "sonic_default": SONIC_DEFAULT_UPPER_BODY,
    "balance_carry": BALANCE_CARRY_UPPER_BODY,
    "soft_horizontal_carry": SOFT_HORIZONTAL_CARRY_UPPER_BODY,
    "horizontal_carry": HORIZONTAL_CARRY_UPPER_BODY,
    "parallel_carry": HORIZONTAL_CARRY_UPPER_BODY,
    "isaac_initial_carry": HORIZONTAL_CARRY_UPPER_BODY,
}

DEFAULT_LEFT_HAND_JOINTS = (0.0, 0.0, 1.75, -1.57, -1.75, -1.57, -1.75)
DEFAULT_RIGHT_HAND_JOINTS = (0.0, 0.0, -1.75, 1.57, 1.75, 1.57, 1.75)
OPEN_LEFT_HAND_JOINTS = (0.0,) * len(DEX3_HAND_JOINT_NAMES)
OPEN_RIGHT_HAND_JOINTS = (0.0,) * len(DEX3_HAND_JOINT_NAMES)

HAND_POSES = {
    "default": (DEFAULT_LEFT_HAND_JOINTS, DEFAULT_RIGHT_HAND_JOINTS),
    "open": (OPEN_LEFT_HAND_JOINTS, OPEN_RIGHT_HAND_JOINTS),
}


def parse_float_tuple(values: Iterable[str] | Iterable[float], *, length: int, name: str) -> tuple[float, ...]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != length:
        raise ValueError(f"{name} must have {length} values, got {len(parsed)}")
    return parsed


def resolve_upper_body_pose(
    pose_name: str,
    custom_values: Iterable[str] | Iterable[float] | None = None,
) -> tuple[float, ...] | None:
    if pose_name == "none":
        if custom_values is not None:
            raise ValueError("--upper-body-values requires --upper-body-pose custom")
        return None
    if pose_name == "custom":
        if custom_values is None:
            raise ValueError("--upper-body-pose custom requires --upper-body-values")
        return parse_float_tuple(custom_values, length=len(UPPER_BODY_JOINT_NAMES), name="upper_body_values")
    if custom_values is not None:
        raise ValueError("--upper-body-values can only be used with --upper-body-pose custom")
    try:
        return UPPER_BODY_POSES[pose_name]
    except KeyError as exc:
        choices = ", ".join(sorted(("none", "custom", *UPPER_BODY_POSES)))
        raise ValueError(f"unknown upper body pose {pose_name!r}; expected one of: {choices}") from exc


def resolve_hand_pose(
    pose_name: str,
    left_values: Iterable[str] | Iterable[float] | None = None,
    right_values: Iterable[str] | Iterable[float] | None = None,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    if pose_name == "none":
        if left_values is not None or right_values is not None:
            raise ValueError("--left-hand-values/--right-hand-values require --hand-pose custom")
        return None, None
    if pose_name == "custom":
        if left_values is None or right_values is None:
            raise ValueError("--hand-pose custom requires both --left-hand-values and --right-hand-values")
        return (
            parse_float_tuple(left_values, length=len(DEX3_HAND_JOINT_NAMES), name="left_hand_values"),
            parse_float_tuple(right_values, length=len(DEX3_HAND_JOINT_NAMES), name="right_hand_values"),
        )
    if left_values is not None or right_values is not None:
        raise ValueError("--left-hand-values/--right-hand-values can only be used with --hand-pose custom")
    try:
        return HAND_POSES[pose_name]
    except KeyError as exc:
        choices = ", ".join(sorted(("none", "custom", *HAND_POSES)))
        raise ValueError(f"unknown hand pose {pose_name!r}; expected one of: {choices}") from exc
