"""Upper-body and hand pose presets for GEAR-SONIC planner commands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


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

# Conservative initial target for a payload-support pose. In the G1 MJCF/SONIC
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

# Arm-only box support target validated with MuJoCo FK on the G1 MJCF:
# the elbow-to-wrist segment is nearly horizontal, the palm stays in front of
# the torso, and the left/right hands are height-symmetric. This is meant for
# the motor-side pose lock, not as evidence of a completed payload experiment.
BOX_SUPPORT_PARALLEL_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.50,
    -0.50,
    0.12,
    -0.12,
    0.0,
    0.0,
    0.40,
    0.40,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

HORIZONTAL_CARRY_UPPER_BODY = BOX_SUPPORT_PARALLEL_UPPER_BODY

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

# User-validated compact carry pose. The name describes the approximate loaded
# shoulder-elbow-wrist included angle, not the MuJoCo elbow coordinate itself.
ELBOW70_CARRY_READY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.55,
    -0.55,
    0.12,
    -0.12,
    0.0,
    0.0,
    -0.45,
    -0.45,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

# Current experiment carry pose. The elbow coordinate remains -0.60 rad while
# the shoulder pitch keeps the forearm nearly level in the settled SONIC pose.
# Waist targets stay zero, and runtime arm transitions exclude waist and legs.
ELBOW60_PAYLOAD_RECOVERY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.10,
    -0.10,
    0.12,
    -0.12,
    0.0,
    0.0,
    -0.60,
    -0.60,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

# Current 60-degree hand carry target. Shoulder and elbow values preserve the
# validated upright elbow60 arm posture. The negative wrist-pitch seed starts
# both palms toward the raised backstop orientation; the MuJoCo runtime closes
# the remaining error from the live palm collision-surface normal.
ELBOW60_HAND60_CARRY_READY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.10,
    -0.10,
    0.12,
    -0.12,
    0.0,
    0.0,
    -0.60,
    -0.60,
    0.0,
    0.0,
    -0.50,
    -0.50,
    0.0,
    0.0,
)

# One-way recovery target. Relative FK moves each support plate toward the torso
# and slightly upward so an outward-slipping payload can settle farther inboard.
ELBOW50_PAYLOAD_RECOVERY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.45,
    -0.45,
    0.12,
    -0.12,
    0.0,
    0.0,
    -0.75,
    -0.75,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

# Earlier carry-ready candidates are retained for explicit pose ablations.
ELBOW110_CARRY_READY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.75,
    -0.75,
    0.12,
    -0.12,
    0.0,
    0.0,
    0.15,
    0.15,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

# Higher carry-ready candidates derived from the same user-validated shoulder /
# elbow sweep as ``elbow110_carry_ready``.  The names describe the measured
# shoulder-elbow-wrist included angle in the loaded G1 MJCF approximately; the
# individual MuJoCo elbow coordinate is not itself 120 or 130 degrees.  Waist
# targets remain zero so these presets cannot introduce a torso crouch.
ELBOW120_CARRY_READY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.80,
    -0.80,
    0.12,
    -0.12,
    0.0,
    0.0,
    0.30,
    0.30,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

ELBOW125_CARRY_READY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.825,
    -0.825,
    0.12,
    -0.12,
    0.0,
    0.0,
    0.375,
    0.375,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

ELBOW130_CARRY_READY_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.85,
    -0.85,
    0.12,
    -0.12,
    0.0,
    0.0,
    0.45,
    0.45,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)

# Keep the selected pose centralized so command metadata, Agent-facing SDK
# docs, and the motor runtime cannot silently diverge.
DEFAULT_CARRY_READY_POSE_NAME = "elbow60_hand60_carry_ready"
PAYLOAD_RECOVERY_POSE_NAME = "elbow50_payload_recovery"
PAYLOAD_RECOVERY_OBSERVATION_S = 15.0


def is_payload_recovery_runtime_event(payload: Mapping[str, object]) -> bool:
    """Require the explicit event field; phase labels are descriptive only."""

    return str(payload.get("upper_body_event", "")) == "recover_payload_inward"

# Final placement target for the motor-side lock. Relative to the carry pose,
# both palms move down and back toward the torso while preserving
# approximately the same left/right span.  This lets a physically supported
# payload settle onto the target before the robots walk away; it is not used
# during transport.
PAYLOAD_RELEASE_UPPER_BODY = (
    0.0,
    0.0,
    0.0,
    -0.10,
    -0.10,
    -0.10,
    0.10,
    0.0,
    0.0,
    0.90,
    0.90,
    0.0,
    0.0,
    0.10,
    0.10,
    0.0,
    0.0,
)


def payload_release_joint_targets(
    joint_names: Iterable[str],
    start_targets: Iterable[float],
    progress: float,
) -> tuple[float, ...]:
    """Open the two supporting arms sideways before folding them down."""

    names = tuple(str(name) for name in joint_names)
    start = tuple(float(value) for value in start_targets)
    if len(names) != len(start):
        raise ValueError("release joint names and start targets must have equal length")

    final_by_name = dict(zip(UPPER_BODY_JOINT_NAMES, PAYLOAD_RELEASE_UPPER_BODY))
    open_by_name = dict(zip(names, start))
    # In the side-by-side carry layout, front's left hand and back's right hand
    # are the inward supports. Move those shoulder rolls away from the payload
    # before elbow flexion can impart a body-forward launch.
    open_by_name["left_shoulder_roll"] = -0.35
    open_by_name["right_shoulder_roll"] = 0.35

    phase = min(1.0, max(0.0, float(progress)))
    if phase <= 0.5:
        alpha = _smoothstep(phase / 0.5)
        return tuple(
            (1.0 - alpha) * start_value
            + alpha * float(open_by_name.get(name, start_value))
            for name, start_value in zip(names, start)
        )

    alpha = _smoothstep((phase - 0.5) / 0.5)
    return tuple(
        (1.0 - alpha) * float(open_by_name.get(name, start_value))
        + alpha * float(final_by_name.get(name, start_value))
        for name, start_value in zip(names, start)
    )


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


UPPER_BODY_POSES = {
    "zero": ZERO_UPPER_BODY,
    "sonic_default": SONIC_DEFAULT_UPPER_BODY,
    "balance_carry": BALANCE_CARRY_UPPER_BODY,
    "elbow70_carry_ready": ELBOW70_CARRY_READY_UPPER_BODY,
    "carry_elbow70": ELBOW70_CARRY_READY_UPPER_BODY,
    "elbow60_payload_recovery": ELBOW60_PAYLOAD_RECOVERY_UPPER_BODY,
    "payload_recovery_elbow60": ELBOW60_PAYLOAD_RECOVERY_UPPER_BODY,
    "elbow60_hand60_carry_ready": ELBOW60_HAND60_CARRY_READY_UPPER_BODY,
    "carry_elbow60_hand60": ELBOW60_HAND60_CARRY_READY_UPPER_BODY,
    "elbow50_payload_recovery": ELBOW50_PAYLOAD_RECOVERY_UPPER_BODY,
    "payload_recovery_elbow50": ELBOW50_PAYLOAD_RECOVERY_UPPER_BODY,
    "elbow110_carry_ready": ELBOW110_CARRY_READY_UPPER_BODY,
    "carry_elbow110": ELBOW110_CARRY_READY_UPPER_BODY,
    "elbow120_carry_ready": ELBOW120_CARRY_READY_UPPER_BODY,
    "carry_elbow120": ELBOW120_CARRY_READY_UPPER_BODY,
    "elbow125_carry_ready": ELBOW125_CARRY_READY_UPPER_BODY,
    "carry_elbow125": ELBOW125_CARRY_READY_UPPER_BODY,
    "elbow130_carry_ready": ELBOW130_CARRY_READY_UPPER_BODY,
    "carry_elbow130": ELBOW130_CARRY_READY_UPPER_BODY,
    "payload_release": PAYLOAD_RELEASE_UPPER_BODY,
    "box_support_parallel": BOX_SUPPORT_PARALLEL_UPPER_BODY,
    "soft_horizontal_carry": SOFT_HORIZONTAL_CARRY_UPPER_BODY,
    "horizontal_carry": HORIZONTAL_CARRY_UPPER_BODY,
    "parallel_box_support": BOX_SUPPORT_PARALLEL_UPPER_BODY,
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
