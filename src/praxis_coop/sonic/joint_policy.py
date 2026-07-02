"""Joint-level SONIC policy contracts for IsaacLab integration.

This module mirrors the public constants and action mapping used by
GEAR-SONIC's C++ deploy path. It does not fake locomotion. Its job is to make
the policy output contract explicit on the Python/IsaacLab side:

    obs_dict -> action[29] -> G1 motor joint position targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


G1_NUM_MOTOR = 29

G1_JOINT_NAMES_MUJOCO_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Kept with the same names as the official C++ header for easier auditing.
# In CreatePolicyCommand(), the policy action is indexed as:
#     action_buffer[ISAACLAB_TO_MUJOCO[mujoco_joint_index]]
# before being written to the MuJoCo/hardware ordered motor target.
ISAACLAB_TO_MUJOCO: tuple[int, ...] = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)

MUJOCO_TO_ISAACLAB: tuple[int, ...] = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)

DEFAULT_ANGLES_MUJOCO_ORDER: tuple[float, ...] = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ * NATURAL_FREQ


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


STIFFNESS_5020 = _stiffness(ARMATURE_5020)
STIFFNESS_7520_14 = _stiffness(ARMATURE_7520_14)
STIFFNESS_7520_22 = _stiffness(ARMATURE_7520_22)
STIFFNESS_4010 = _stiffness(ARMATURE_4010)
DAMPING_5020 = _damping(ARMATURE_5020)
DAMPING_7520_14 = _damping(ARMATURE_7520_14)
DAMPING_7520_22 = _damping(ARMATURE_7520_22)
DAMPING_4010 = _damping(ARMATURE_4010)

ACTION_SCALE_MUJOCO_ORDER: tuple[float, ...] = (
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,
)

KP_MUJOCO_ORDER: tuple[float, ...] = (
    STIFFNESS_7520_22,
    STIFFNESS_7520_22,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,
    STIFFNESS_7520_22,
    STIFFNESS_7520_22,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,
    STIFFNESS_7520_14,
    2.0 * STIFFNESS_5020,
    2.0 * STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_4010,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_5020,
    STIFFNESS_4010,
    STIFFNESS_4010,
)

KD_MUJOCO_ORDER: tuple[float, ...] = (
    DAMPING_7520_22,
    DAMPING_7520_22,
    DAMPING_7520_14,
    DAMPING_7520_22,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,
    DAMPING_7520_22,
    DAMPING_7520_22,
    DAMPING_7520_14,
    DAMPING_7520_22,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,
    DAMPING_7520_14,
    2.0 * DAMPING_5020,
    2.0 * DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_4010,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_5020,
    DAMPING_4010,
    DAMPING_4010,
)


@dataclass(frozen=True)
class SonicMotorCommand:
    """Joint PD command matching GEAR-SONIC's C++ MotorCommand layout."""

    joint_names: tuple[str, ...]
    q_target: tuple[float, ...]
    dq_target: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    tau_ff: tuple[float, ...]
    order: str = "mujoco_hardware"

    def to_json(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "q_target": list(self.q_target),
            "dq_target": list(self.dq_target),
            "kp": list(self.kp),
            "kd": list(self.kd),
            "tau_ff": list(self.tau_ff),
            "order": self.order,
        }


@dataclass(frozen=True)
class SonicIsaacJointBackend:
    """Apply official SONIC 29-dof policy actions to IsaacLab G1 articulations."""

    write_gains: bool = False

    def action_to_motor_command(self, action: Sequence[float]) -> SonicMotorCommand:
        return sonic_action_to_motor_command(action)

    def zero_motor_command(self) -> SonicMotorCommand:
        return sonic_action_to_motor_command(zero_policy_action())

    def resolve_joint_ids(self, robot: Any) -> list[int]:
        return resolve_isaac_joint_ids(robot)

    def apply_action(self, robot: Any, action: Sequence[float]) -> SonicMotorCommand:
        command = self.action_to_motor_command(action)
        apply_motor_command_to_isaac_articulation(robot, command, write_gains=self.write_gains)
        return command

    def apply_zero_action(self, robot: Any) -> SonicMotorCommand:
        command = self.zero_motor_command()
        apply_motor_command_to_isaac_articulation(robot, command, write_gains=self.write_gains)
        return command


@dataclass(frozen=True)
class OnnxTensorInfo:
    name: str
    shape: tuple[int | str, ...]
    elem_type: int

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "elem_type": self.elem_type}


@dataclass(frozen=True)
class OnnxModelInfo:
    path: str
    inputs: tuple[OnnxTensorInfo, ...]
    outputs: tuple[OnnxTensorInfo, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "inputs": [item.to_json() for item in self.inputs],
            "outputs": [item.to_json() for item in self.outputs],
        }


def validate_sonic_constants() -> None:
    """Fail early if copied policy constants are internally inconsistent."""

    arrays = {
        "G1_JOINT_NAMES_MUJOCO_ORDER": G1_JOINT_NAMES_MUJOCO_ORDER,
        "ISAACLAB_TO_MUJOCO": ISAACLAB_TO_MUJOCO,
        "MUJOCO_TO_ISAACLAB": MUJOCO_TO_ISAACLAB,
        "DEFAULT_ANGLES_MUJOCO_ORDER": DEFAULT_ANGLES_MUJOCO_ORDER,
        "ACTION_SCALE_MUJOCO_ORDER": ACTION_SCALE_MUJOCO_ORDER,
        "KP_MUJOCO_ORDER": KP_MUJOCO_ORDER,
        "KD_MUJOCO_ORDER": KD_MUJOCO_ORDER,
    }
    for name, value in arrays.items():
        if len(value) != G1_NUM_MOTOR:
            raise ValueError(f"{name} must have {G1_NUM_MOTOR} entries, got {len(value)}")
    if sorted(ISAACLAB_TO_MUJOCO) != list(range(G1_NUM_MOTOR)):
        raise ValueError("ISAACLAB_TO_MUJOCO must be a permutation of 0..28")
    if sorted(MUJOCO_TO_ISAACLAB) != list(range(G1_NUM_MOTOR)):
        raise ValueError("MUJOCO_TO_ISAACLAB must be a permutation of 0..28")


def zero_policy_action() -> tuple[float, ...]:
    return (0.0,) * G1_NUM_MOTOR


def sonic_action_to_motor_command(action: Sequence[float]) -> SonicMotorCommand:
    """Map official SONIC action[29] to hardware-ordered motor PD targets.

    This intentionally mirrors:

        action_value = action_buffer[isaaclab_to_mujoco[i]] * g1_action_scale[i]
        q_target[i] = default_angles[i] + action_value

    from GEAR-SONIC's ``CreatePolicyCommand``.
    """

    validate_sonic_constants()
    if len(action) != G1_NUM_MOTOR:
        raise ValueError(f"SONIC action must have {G1_NUM_MOTOR} values, got {len(action)}")
    action_values = tuple(float(value) for value in action)
    q_target = tuple(
        DEFAULT_ANGLES_MUJOCO_ORDER[i] + action_values[ISAACLAB_TO_MUJOCO[i]] * ACTION_SCALE_MUJOCO_ORDER[i]
        for i in range(G1_NUM_MOTOR)
    )
    return SonicMotorCommand(
        joint_names=G1_JOINT_NAMES_MUJOCO_ORDER,
        q_target=q_target,
        dq_target=(0.0,) * G1_NUM_MOTOR,
        kp=KP_MUJOCO_ORDER,
        kd=KD_MUJOCO_ORDER,
        tau_ff=(0.0,) * G1_NUM_MOTOR,
    )


def load_onnx_model_info(path: str | Path) -> OnnxModelInfo:
    """Read ONNX tensor names and shapes without requiring onnxruntime."""

    import onnx

    model_path = Path(path)
    model = onnx.load(str(model_path), load_external_data=False)
    return OnnxModelInfo(
        path=str(model_path),
        inputs=tuple(_tensor_info(item) for item in model.graph.input),
        outputs=tuple(_tensor_info(item) for item in model.graph.output),
    )


def make_isaac_joint_target_tensor(robot: Any, command: SonicMotorCommand):
    """Return ``(target, joint_ids)`` for an IsaacLab articulation.

    The returned target is shaped ``(num_envs, 29)`` and ordered exactly like the
    returned ``joint_ids``. It is suitable for
    ``robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)``.
    """

    import torch

    joint_ids = resolve_isaac_joint_ids(robot, command.joint_names)
    q_target = torch.tensor(command.q_target, dtype=torch.float32, device=robot.device)
    target = q_target.unsqueeze(0).repeat(robot.num_instances, 1)
    return target, joint_ids


def apply_motor_command_to_isaac_articulation(
    robot: Any,
    command: SonicMotorCommand,
    *,
    write_gains: bool = False,
) -> None:
    """Apply a SONIC motor command as IsaacLab joint targets.

    This sets targets, not joint state. It therefore stays on the controller
    path and does not teleport the robot.
    """

    import torch

    target, joint_ids = make_isaac_joint_target_tensor(robot, command)
    robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)
    zero_vel = torch.zeros_like(target)
    robot.set_joint_velocity_target_index(target=zero_vel, joint_ids=joint_ids)
    if write_gains:
        kp = torch.tensor(command.kp, dtype=torch.float32, device=robot.device).unsqueeze(0).repeat(robot.num_instances, 1)
        kd = torch.tensor(command.kd, dtype=torch.float32, device=robot.device).unsqueeze(0).repeat(robot.num_instances, 1)
        robot.write_joint_stiffness_to_sim_index(stiffness=kp, joint_ids=joint_ids)
        robot.write_joint_damping_to_sim_index(damping=kd, joint_ids=joint_ids)


def resolve_isaac_joint_ids(robot: Any, joint_names: Sequence[str] = G1_JOINT_NAMES_MUJOCO_ORDER) -> list[int]:
    """Resolve SONIC joint names against an IsaacLab articulation."""

    available = list(robot.joint_names)
    name_to_id = {name: idx for idx, name in enumerate(available)}
    missing = [name for name in joint_names if name not in name_to_id]
    if missing:
        raise ValueError(
            "Isaac articulation is missing SONIC G1 joints: "
            + ", ".join(missing)
            + f". Available joints: {available}"
        )
    return [name_to_id[name] for name in joint_names]


def _tensor_info(value_info: Any) -> OnnxTensorInfo:
    dims: list[int | str] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append("?")
    return OnnxTensorInfo(name=value_info.name, shape=tuple(dims), elem_type=int(tensor_type.elem_type))


validate_sonic_constants()
