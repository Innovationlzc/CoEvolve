"""Typed parameter patches for Stage 1 straight-carry evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from praxis_coop.models import PatchProposal


@dataclass
class Stage1Parameters:
    warmup_steps: int = 30
    path_speed: float = 0.08
    forward_vy: float = 0.22
    robot_a_vx: float = 0.0
    robot_a_vy: float | None = None
    robot_a_wz: float = 0.0
    robot_b_vx: float = 0.0
    robot_b_vy: float | None = None
    robot_b_wz: float = 0.0
    lower_mode: str = "constant"
    root_kp: float = 0.65
    root_max_vel: float = 0.20
    stand_off_x: float = 0.32
    hip_height: float = 0.74
    wrist_z: float = 0.88
    wrist_x_offset: float = 0.08
    wrist_y_halfspan: float = 0.16
    hand_target: float = 0.35
    wrist_orientation: str = "box_yaw"
    upper_mode: str = "hold_initial_scene"

    def to_cli_args(self) -> list[str]:
        payload = asdict(self)
        args: list[str] = []
        for key, value in payload.items():
            if value is None:
                continue
            args.extend([f"--{key.replace('_', '-')}", str(value)])
        return args

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage1SonicParameters:
    """Bounded high-level SONIC command parameters for Stage 1 evolution.

    These are the only values Qwen may patch in the real Stage1 loop.  They
    shape the SONIC planner command; they do not expose Isaac root/object state
    writes or task-level action adapters.
    """

    warmup_steps: int = 20
    target_vel: float = 0.08
    robot_a_target_vel: float | None = None
    robot_b_target_vel: float | None = None
    robot_a_movement_x: float = 0.0
    robot_a_movement_y: float = 1.0
    robot_a_movement_z: float = 0.0
    robot_b_movement_x: float = 0.0
    robot_b_movement_y: float = 1.0
    robot_b_movement_z: float = 0.0
    robot_a_facing_x: float = 1.0
    robot_a_facing_y: float = 0.0
    robot_a_facing_z: float = 0.0
    robot_b_facing_x: float = 1.0
    robot_b_facing_y: float = 0.0
    robot_b_facing_z: float = 0.0
    replan_interval_steps: int = 10
    default_height: float = 0.78
    sonic_height: float = -1.0
    mode: str = "object_carrying"

    def to_cli_args(self) -> list[str]:
        args: list[str] = [
            "--warmup-steps",
            str(self.warmup_steps),
            "--target-vel",
            str(self.target_vel),
            "--sonic-height",
            str(self.sonic_height),
            "--replan-interval-steps",
            str(self.replan_interval_steps),
            "--default-height",
            str(self.default_height),
            "--mode",
            str(self.mode),
            "--robot-a-movement",
            str(self.robot_a_movement_x),
            str(self.robot_a_movement_y),
            str(self.robot_a_movement_z),
            "--robot-b-movement",
            str(self.robot_b_movement_x),
            str(self.robot_b_movement_y),
            str(self.robot_b_movement_z),
            "--robot-a-facing",
            str(self.robot_a_facing_x),
            str(self.robot_a_facing_y),
            str(self.robot_a_facing_z),
            "--robot-b-facing",
            str(self.robot_b_facing_x),
            str(self.robot_b_facing_y),
            str(self.robot_b_facing_z),
        ]
        if self.robot_a_target_vel is not None:
            args.extend(["--robot-a-target-vel", str(self.robot_a_target_vel)])
        if self.robot_b_target_vel is not None:
            args.extend(["--robot-b-target-vel", str(self.robot_b_target_vel)])
        return args

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "warmup_steps": (0, 120),
    "path_speed": (0.02, 0.18),
    "forward_vy": (0.02, 0.35),
    "robot_a_vx": (-0.12, 0.12),
    "robot_a_vy": (-0.35, 0.35),
    "robot_a_wz": (-0.25, 0.25),
    "robot_b_vx": (-0.12, 0.12),
    "robot_b_vy": (-0.35, 0.35),
    "robot_b_wz": (-0.25, 0.25),
    "root_kp": (0.15, 1.20),
    "root_max_vel": (0.05, 0.35),
    "stand_off_x": (0.20, 0.55),
    "hip_height": (0.62, 0.86),
}

ENUM_VALUES: dict[str, set[str]] = {
    "lower_mode": {"constant", "planned_follow"},
}

PATCHABLE_FIELDS = set(PARAMETER_BOUNDS) | set(ENUM_VALUES)

SONIC_PARAMETER_BOUNDS: dict[str, tuple[float, float]] = {
    "warmup_steps": (0, 120),
    "target_vel": (0.0, 0.22),
    "robot_a_target_vel": (0.0, 0.22),
    "robot_b_target_vel": (0.0, 0.22),
    "robot_a_movement_x": (-1.0, 1.0),
    "robot_a_movement_y": (-1.0, 1.0),
    "robot_a_movement_z": (-0.1, 0.1),
    "robot_b_movement_x": (-1.0, 1.0),
    "robot_b_movement_y": (-1.0, 1.0),
    "robot_b_movement_z": (-0.1, 0.1),
    "robot_a_facing_x": (-1.0, 1.0),
    "robot_a_facing_y": (-1.0, 1.0),
    "robot_a_facing_z": (-0.1, 0.1),
    "robot_b_facing_x": (-1.0, 1.0),
    "robot_b_facing_y": (-1.0, 1.0),
    "robot_b_facing_z": (-0.1, 0.1),
    "replan_interval_steps": (2, 40),
    "default_height": (0.68, 0.86),
    "sonic_height": (-1.0, 0.86),
}

SONIC_ENUM_VALUES: dict[str, set[str]] = {
    "mode": {"object_carrying", "slow_walk", "walk"},
}

SONIC_NULLABLE_FIELDS = {"robot_a_target_vel", "robot_b_target_vel"}
SONIC_INTEGER_FIELDS = {"warmup_steps", "replan_interval_steps"}
SONIC_PATCHABLE_FIELDS = set(SONIC_PARAMETER_BOUNDS) | set(SONIC_ENUM_VALUES)


def apply_stage1_patch(params: Stage1Parameters, patch: PatchProposal) -> Stage1Parameters:
    """Apply a bounded Stage1 parameter patch returned by Qwen."""

    if patch.patch_type != "parameter":
        raise ValueError(f"Stage1 only accepts parameter patches, got {patch.patch_type!r}")
    if patch.target not in {"stage1_parameters", "stage1_parameters.parameters"}:
        raise ValueError(f"Unsupported Stage1 patch target: {patch.target!r}")

    values = params.to_json()
    for key, raw_value in patch.change.items():
        if key not in PATCHABLE_FIELDS:
            raise ValueError(f"Qwen attempted to patch unsupported Stage1 parameter: {key}")
        if key in ENUM_VALUES:
            value = str(raw_value)
            if value not in ENUM_VALUES[key]:
                allowed = ", ".join(sorted(ENUM_VALUES[key]))
                raise ValueError(f"Invalid {key}={value!r}; allowed: {allowed}")
            values[key] = value
            continue

        if raw_value is None and key in {"robot_a_vy", "robot_b_vy"}:
            values[key] = None
            continue
        low, high = PARAMETER_BOUNDS[key]
        value = float(raw_value)
        value = min(max(value, low), high)
        values[key] = int(round(value)) if key == "warmup_steps" else value

    return Stage1Parameters(**values)


def apply_stage1_sonic_patch(params: Stage1SonicParameters, patch: PatchProposal) -> Stage1SonicParameters:
    """Apply a bounded Qwen patch to Stage1 SONIC command parameters."""

    if patch.patch_type != "parameter":
        raise ValueError(f"Stage1 SONIC only accepts parameter patches, got {patch.patch_type!r}")
    if patch.target not in {"stage1_sonic_parameters", "stage1_sonic_parameters.parameters"}:
        raise ValueError(f"Unsupported Stage1 SONIC patch target: {patch.target!r}")

    values = params.to_json()
    for key, raw_value in patch.change.items():
        if key not in SONIC_PATCHABLE_FIELDS:
            raise ValueError(f"Qwen attempted to patch unsupported Stage1 SONIC parameter: {key}")
        if key in SONIC_ENUM_VALUES:
            value = str(raw_value)
            if value not in SONIC_ENUM_VALUES[key]:
                allowed = ", ".join(sorted(SONIC_ENUM_VALUES[key]))
                raise ValueError(f"Invalid {key}={value!r}; allowed: {allowed}")
            values[key] = value
            continue
        if raw_value is None and key in SONIC_NULLABLE_FIELDS:
            values[key] = None
            continue

        low, high = SONIC_PARAMETER_BOUNDS[key]
        value = float(raw_value)
        value = min(max(value, low), high)
        values[key] = int(round(value)) if key in SONIC_INTEGER_FIELDS else value

    return Stage1SonicParameters(**values)


def stage1_patch_schema() -> dict[str, Any]:
    return {
        "target": "stage1_parameters",
        "patch_type": "parameter",
        "patchable_fields": sorted(PATCHABLE_FIELDS),
        "numeric_bounds": PARAMETER_BOUNDS,
        "enum_values": {key: sorted(values) for key, values in ENUM_VALUES.items()},
    }


def stage1_sonic_patch_schema() -> dict[str, Any]:
    return {
        "target": "stage1_sonic_parameters",
        "patch_type": "parameter",
        "patchable_fields": sorted(SONIC_PATCHABLE_FIELDS),
        "numeric_bounds": SONIC_PARAMETER_BOUNDS,
        "enum_values": {key: sorted(values) for key, values in SONIC_ENUM_VALUES.items()},
        "nullable_fields": sorted(SONIC_NULLABLE_FIELDS),
        "locked_contract": {
            "upper_body_pose_lock": True,
            "hand_joint_pose_lock": True,
            "env_action_surface_used": False,
            "direct_root_or_box_write": False,
            "proxy_action_used": False,
        },
    }
