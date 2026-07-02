"""Python ONNX runtime bridge for SONIC release policy in IsaacLab.

The official GEAR-SONIC deploy loop is C++/TensorRT.  This module mirrors the
same high-level data path on the Python side so an IsaacLab task can be driven
through joint targets instead of task-level proxy actions:

    planner command -> planner qpos -> encoder token -> decoder action[29].

The resulting action is still applied by :mod:`praxis_coop.sonic.joint_policy`.
No root pose, object pose, or object velocity is written here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from praxis_coop.sonic.commands import SonicLocomotionMode
from praxis_coop.sonic.joint_policy import (
    DEFAULT_ANGLES_MUJOCO_ORDER,
    G1_JOINT_NAMES_MUJOCO_ORDER,
    G1_NUM_MOTOR,
    MUJOCO_TO_ISAACLAB,
    resolve_isaac_joint_ids,
)


POLICY_OBS_DIM = 994
ENCODER_OBS_DIM = 1762
TOKEN_DIM = 64
PLANNER_QPOS_DIM = 36
PLANNER_CONTEXT_FRAMES = 4
PLANNER_OUTPUT_FRAMES = 64
SONIC_HISTORY_FRAMES = 10
SONIC_CONTROL_HZ = 50.0
PLANNER_OUTPUT_HZ = 30.0


@dataclass(frozen=True)
class SonicPlannerCommand:
    """Single robot locomotion command for the SONIC planner."""

    mode: int = int(SonicLocomotionMode.IDLE)
    target_vel: float = -1.0
    height: float = -1.0
    movement: tuple[float, float, float] = (0.0, 0.0, 0.0)
    facing: tuple[float, float, float] = (1.0, 0.0, 0.0)
    random_seed: int = 0


@dataclass(frozen=True)
class SonicPlannerMotion:
    """50 Hz reference motion extracted from planner ``mujoco_qpos`` output."""

    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    num_pred_frames_30hz: int

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


class SonicOnnxRuntimeUnavailable(RuntimeError):
    """Raised when policy inference is requested without ONNX Runtime."""


class SonicReleaseOnnx:
    """Shared ONNX Runtime sessions for the release SONIC planner/policy."""

    def __init__(
        self,
        *,
        policy_dir: str | Path,
        planner_onnx: str | Path,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise SonicOnnxRuntimeUnavailable(
                "SONIC policy inference requires onnxruntime in the IsaacLab Python env. "
                "Install it with: /home/admin1/Project/IsaacLab/env_isaaclab/bin/python -m pip install onnxruntime"
            ) from exc

        policy_path = Path(policy_dir)
        self.encoder_path = policy_path / "model_encoder.onnx"
        self.decoder_path = policy_path / "model_decoder.onnx"
        self.planner_path = Path(planner_onnx)
        for path in (self.encoder_path, self.decoder_path, self.planner_path):
            if not path.exists():
                raise FileNotFoundError(path)

        available = set(ort.get_available_providers())
        requested = list(providers or ("CUDAExecutionProvider", "CPUExecutionProvider"))
        selected = [provider for provider in requested if provider in available]
        if not selected:
            selected = ["CPUExecutionProvider"]

        self.providers = selected
        self.encoder = ort.InferenceSession(str(self.encoder_path), providers=selected)
        self.decoder = ort.InferenceSession(str(self.decoder_path), providers=selected)
        self.planner = ort.InferenceSession(str(self.planner_path), providers=selected)

    def infer_planner(self, context_qpos: np.ndarray, command: SonicPlannerCommand) -> tuple[np.ndarray, int]:
        """Run planner ONNX and return ``(mujoco_qpos[64,36], num_pred_frames)``."""

        inputs = {
            "context_mujoco_qpos": _batched_float(context_qpos, (PLANNER_CONTEXT_FRAMES, PLANNER_QPOS_DIM)),
            "target_vel": np.asarray([command.target_vel], dtype=np.float32),
            "mode": np.asarray([command.mode], dtype=np.int64),
            "movement_direction": _batched_float(command.movement, (3,)),
            "facing_direction": _batched_float(command.facing, (3,)),
            "random_seed": np.asarray([command.random_seed], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": np.asarray([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64),
            "height": np.asarray([command.height], dtype=np.float32),
        }
        qpos, num_pred = self.planner.run(["mujoco_qpos", "num_pred_frames"], inputs)
        qpos = np.asarray(qpos, dtype=np.float32).reshape(PLANNER_OUTPUT_FRAMES, PLANNER_QPOS_DIM)
        return qpos, int(np.asarray(num_pred).reshape(-1)[0])

    def infer_token(self, encoder_obs: np.ndarray) -> np.ndarray:
        obs = _batched_float(encoder_obs, (ENCODER_OBS_DIM,))
        token = self.encoder.run(["encoded_tokens"], {"obs_dict": obs})[0]
        return np.asarray(token, dtype=np.float32).reshape(TOKEN_DIM)

    def infer_action(self, policy_obs: np.ndarray) -> np.ndarray:
        obs = _batched_float(policy_obs, (POLICY_OBS_DIM,))
        action = self.decoder.run(["action"], {"obs_dict": obs})[0]
        return np.asarray(action, dtype=np.float32).reshape(G1_NUM_MOTOR)


class SonicPlannerRuntime:
    """Small Python equivalent of the C++ LocalMotionPlanner ONNX path."""

    def __init__(self, sessions: SonicReleaseOnnx, *, default_height: float = 0.78) -> None:
        self.sessions = sessions
        self.default_height = float(default_height)
        self.context_qpos = np.zeros((PLANNER_CONTEXT_FRAMES, PLANNER_QPOS_DIM), dtype=np.float32)
        self.last_motion: SonicPlannerMotion | None = None

    def initialize_context(self, *, joint_pos_mujoco: Sequence[float], root_quat_wxyz: Sequence[float] | None = None) -> None:
        # The official C++ InitializeContext normalizes the context to a
        # zero-yaw frame and writes an identity quaternion for all context
        # frames.  Keep the unused argument for call-site readability.
        _ = root_quat_wxyz
        quat = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
        q = np.asarray(joint_pos_mujoco, dtype=np.float32).reshape(G1_NUM_MOTOR)
        for frame in range(PLANNER_CONTEXT_FRAMES):
            self.context_qpos[frame, 0:3] = (0.0, 0.0, self.default_height)
            self.context_qpos[frame, 3:7] = quat
            self.context_qpos[frame, 7:] = q

    def replan(self, command: SonicPlannerCommand) -> SonicPlannerMotion:
        qpos_30hz, num_pred = self.sessions.infer_planner(self.context_qpos, command)
        motion = resample_planner_qpos_to_50hz(qpos_30hz, num_pred)
        self.last_motion = motion
        self._update_context_from_motion(motion)
        return motion

    def _update_context_from_motion(self, motion: SonicPlannerMotion) -> None:
        if motion.num_frames <= 0:
            return
        frame_ids = np.linspace(0, max(0, motion.num_frames - 1), PLANNER_CONTEXT_FRAMES).astype(int)
        for context_idx, frame_idx in enumerate(frame_ids):
            self.context_qpos[context_idx, 0:3] = motion.root_pos[frame_idx]
            self.context_qpos[context_idx, 3:7] = motion.root_quat[frame_idx]
            for joint in range(G1_NUM_MOTOR):
                self.context_qpos[context_idx, 7 + MUJOCO_TO_ISAACLAB[joint]] = motion.joint_pos[frame_idx, joint]


class SonicReleaseObservationBuilder:
    """Build SONIC release decoder observations from a live Isaac articulation."""

    def __init__(self, robot: Any, *, token: Sequence[float] | None = None) -> None:
        self.robot = robot
        self.joint_ids = resolve_isaac_joint_ids(robot)
        self.token = np.asarray(token if token is not None else np.zeros(TOKEN_DIM), dtype=np.float32).reshape(TOKEN_DIM)
        self._base_ang_vel: list[np.ndarray] = []
        self._joint_pos: list[np.ndarray] = []
        self._joint_vel: list[np.ndarray] = []
        self._last_action: list[np.ndarray] = []
        self._gravity: list[np.ndarray] = []
        self.last_action = np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        self.push_robot_state(self.last_action)
        while len(self._base_ang_vel) < SONIC_HISTORY_FRAMES:
            self.push_robot_state(self.last_action)

    def current_joint_pos_mujoco(self) -> np.ndarray:
        return _joint_values_by_ids(self.robot.data.joint_pos, self.joint_ids)

    def current_root_quat_wxyz(self) -> np.ndarray:
        # IsaacLab 3.0 exposes root quaternions as xyzw even though the
        # property name is ``root_quat_w``. SONIC's release C++ path and motion
        # planner use wxyz, so convert at the bridge boundary.
        quat_xyzw = _tensor_first_env(self.robot.data.root_quat_w)
        return _xyzw_to_wxyz(np.asarray(quat_xyzw, dtype=np.float32).reshape(4))

    def push_robot_state(self, last_action: Sequence[float]) -> None:
        q_abs = self.current_joint_pos_mujoco()
        dq_abs = _joint_values_by_ids(self.robot.data.joint_vel, self.joint_ids)
        q_policy = np.asarray(
            [q_abs[MUJOCO_TO_ISAACLAB[i]] - DEFAULT_ANGLES_MUJOCO_ORDER[MUJOCO_TO_ISAACLAB[i]] for i in range(G1_NUM_MOTOR)],
            dtype=np.float32,
        )
        dq_policy = np.asarray([dq_abs[MUJOCO_TO_ISAACLAB[i]] for i in range(G1_NUM_MOTOR)], dtype=np.float32)
        base_ang_vel = _base_angular_velocity(self.robot)
        gravity = _gravity_in_base_frame(self.current_root_quat_wxyz())
        self._append(self._base_ang_vel, base_ang_vel)
        self._append(self._joint_pos, q_policy)
        self._append(self._joint_vel, dq_policy)
        self._append(self._last_action, np.asarray(last_action, dtype=np.float32).reshape(G1_NUM_MOTOR))
        self._append(self._gravity, gravity)
        self.last_action = np.asarray(last_action, dtype=np.float32).reshape(G1_NUM_MOTOR)

    def set_token(self, token: Sequence[float]) -> None:
        self.token = np.asarray(token, dtype=np.float32).reshape(TOKEN_DIM)

    def build_policy_obs(self) -> np.ndarray:
        parts = [
            self.token,
            self._history(self._base_ang_vel, 3),
            self._history(self._joint_pos, G1_NUM_MOTOR),
            self._history(self._joint_vel, G1_NUM_MOTOR),
            self._history(self._last_action, G1_NUM_MOTOR),
            self._history(self._gravity, 3),
        ]
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (POLICY_OBS_DIM,):
            raise ValueError(f"SONIC policy obs must be {POLICY_OBS_DIM}D, got {obs.shape}")
        return obs

    def build_encoder_obs(self, motion: SonicPlannerMotion | None, *, frame: int = 0) -> np.ndarray:
        """Build encoder observations in the official release config layout.

        The release encoder ONNX expects the full 1762D buffer described by
        ``policy/release/observation_config.yaml``.  For G1 planner mode the C++
        deploy fills only these fields and leaves the other enabled encoder
        slots as zero:

        - ``encoder_mode_4`` (4D)
        - ``motion_joint_positions_10frame_step5`` (290D)
        - ``motion_joint_velocities_10frame_step5`` (290D)
        - ``motion_anchor_orientation_10frame_step5`` (60D)

        Keeping the official offsets matters: the encoder is shared across G1,
        teleop, and SMPL tokenizers, so packing only the non-zero G1 fields at
        the start corrupts the token.
        """

        obs = np.zeros(ENCODER_OBS_DIM, dtype=np.float32)
        # encoder_mode_4. G1 mode id is 0; the remaining three values are the
        # same zero padding as GatherEncoderMode(..., fill_zeros_num=3).
        obs[0:4] = 0.0
        if motion is None or motion.num_frames <= 0:
            return obs

        q = _sample_future_frames(motion.joint_pos, frame=frame, num_frames=10, step=5, dim=G1_NUM_MOTOR)
        dq = _sample_future_frames(motion.joint_vel, frame=frame, num_frames=10, step=5, dim=G1_NUM_MOTOR)
        ori = _motion_anchor_orientation_10frame_step5(
            motion,
            frame=frame,
            base_quat_wxyz=self.current_root_quat_wxyz(),
        )

        obs[4:294] = q.reshape(-1)
        obs[294:584] = dq.reshape(-1)
        obs[601:661] = ori.reshape(-1)
        return obs

    def _history(self, values: list[np.ndarray], dim: int) -> np.ndarray:
        if not values:
            return np.zeros(SONIC_HISTORY_FRAMES * dim, dtype=np.float32)
        padded = list(values)
        while len(padded) < SONIC_HISTORY_FRAMES:
            padded.insert(0, padded[0])
        return np.stack(padded[-SONIC_HISTORY_FRAMES:], axis=0).reshape(-1).astype(np.float32)

    @staticmethod
    def _append(values: list[np.ndarray], value: np.ndarray) -> None:
        values.append(np.asarray(value, dtype=np.float32).copy())
        del values[:-SONIC_HISTORY_FRAMES]


class SonicReleasePolicyController:
    """Per-robot controller: planner + encoder + decoder + observation history."""

    def __init__(
        self,
        *,
        robot: Any,
        sessions: SonicReleaseOnnx,
        command: SonicPlannerCommand,
        replan_interval_steps: int = 10,
        default_height: float = 0.78,
    ) -> None:
        self.robot = robot
        self.sessions = sessions
        self.command = command
        self.replan_interval_steps = max(1, int(replan_interval_steps))
        self.obs_builder = SonicReleaseObservationBuilder(robot)
        self.planner = SonicPlannerRuntime(sessions, default_height=default_height)
        self.planner.initialize_context(
            joint_pos_mujoco=self.obs_builder.current_joint_pos_mujoco(),
            root_quat_wxyz=self.obs_builder.current_root_quat_wxyz(),
        )
        self.steps = 0
        self.motion_frame = 0
        self.last_motion: SonicPlannerMotion | None = None

    def set_command(self, command: SonicPlannerCommand, *, force_replan: bool = True) -> None:
        """Update the low-frequency planner command for the next inference step."""

        if command == self.command:
            return
        self.command = command
        if force_replan:
            self.steps = 0

    def infer_action(self) -> np.ndarray:
        # Official deploy logs the current robot state together with the
        # previously issued action before building observations. Do the same
        # here; the newly inferred action becomes "last_action" for the next
        # control tick after physics has stepped.
        self.obs_builder.push_robot_state(self.obs_builder.last_action)
        if self.steps == 0 or self.steps % self.replan_interval_steps == 0:
            self.last_motion = self.planner.replan(self.command)
            self.motion_frame = 0
        encoder_obs = self.obs_builder.build_encoder_obs(self.last_motion, frame=self.motion_frame)
        self.obs_builder.set_token(self.sessions.infer_token(encoder_obs))
        policy_obs = self.obs_builder.build_policy_obs()
        action = self.sessions.infer_action(policy_obs)
        self.obs_builder.last_action = np.asarray(action, dtype=np.float32).reshape(G1_NUM_MOTOR)
        self.steps += 1
        if self.last_motion is not None and self.last_motion.num_frames > 0:
            self.motion_frame = min(self.motion_frame + 1, self.last_motion.num_frames - 1)
        return action


def resample_planner_qpos_to_50hz(qpos_30hz: np.ndarray, num_pred_frames: int) -> SonicPlannerMotion:
    """Mirror C++ ``ResampleGeneratedSequence50Hz`` for planner output."""

    qpos = np.asarray(qpos_30hz, dtype=np.float32).reshape(PLANNER_OUTPUT_FRAMES, PLANNER_QPOS_DIM)
    frames_30 = max(2, min(int(num_pred_frames), qpos.shape[0]))
    frames_50 = max(2, int(np.floor((frames_30 / PLANNER_OUTPUT_HZ) * SONIC_CONTROL_HZ)))
    root_pos = np.zeros((frames_50, 3), dtype=np.float32)
    root_quat = np.zeros((frames_50, 4), dtype=np.float32)
    joint_pos = np.zeros((frames_50, G1_NUM_MOTOR), dtype=np.float32)

    for frame_50 in range(frames_50):
        t = frame_50 / SONIC_CONTROL_HZ
        frame_30 = t * PLANNER_OUTPUT_HZ
        f0 = min(int(np.floor(frame_30)), frames_30 - 1)
        f1 = min(f0 + 1, frames_30 - 1)
        w1 = frame_30 - f0
        w0 = 1.0 - w1
        root_pos[frame_50] = w0 * qpos[f0, 0:3] + w1 * qpos[f1, 0:3]
        root_quat[frame_50] = _normalize_quat(w0 * qpos[f0, 3:7] + w1 * qpos[f1, 3:7])
        for joint in range(G1_NUM_MOTOR):
            slot = 7 + MUJOCO_TO_ISAACLAB[joint]
            joint_pos[frame_50, joint] = w0 * qpos[f0, slot] + w1 * qpos[f1, slot]

    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) * SONIC_CONTROL_HZ
    joint_vel[-1] = joint_vel[-2]
    return SonicPlannerMotion(root_pos=root_pos, root_quat=root_quat, joint_pos=joint_pos, joint_vel=joint_vel, num_pred_frames_30hz=frames_30)


def _batched_float(value: Any, inner_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == inner_shape:
        arr = arr.reshape((1, *inner_shape))
    expected = (1, *inner_shape)
    if arr.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {arr.shape}")
    return arr


def _joint_values_by_ids(value: Any, joint_ids: Sequence[int]) -> np.ndarray:
    tensor = _tensor_first_env(value)
    return np.asarray(tensor[list(joint_ids)], dtype=np.float32)


def _tensor_first_env(value: Any) -> np.ndarray:
    tensor = getattr(value, "torch", value)
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    arr = np.asarray(tensor, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr[0] if arr.ndim >= 2 else arr


def _base_angular_velocity(robot: Any) -> np.ndarray:
    for name in ("root_ang_vel_b", "root_ang_vel_w"):
        if hasattr(robot.data, name):
            return np.asarray(_tensor_first_env(getattr(robot.data, name)), dtype=np.float32).reshape(3)
    return np.zeros(3, dtype=np.float32)


def _gravity_in_base_frame(root_quat_wxyz: Sequence[float]) -> np.ndarray:
    return _quat_rotate(_quat_conjugate(np.asarray(root_quat_wxyz, dtype=np.float32)), np.asarray((0.0, 0.0, -1.0), dtype=np.float32))


def _quat_sequence_to_6d(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.stack([_quat_to_6d(quat) for quat in quat_wxyz], axis=0).astype(np.float32)


def _sample_future_frames(values: np.ndarray, *, frame: int, num_frames: int, step: int, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, dim)
    if arr.shape[0] == 0:
        return np.zeros((num_frames, dim), dtype=np.float32)
    start = max(0, int(frame))
    sampled = []
    for idx in range(num_frames):
        frame_idx = min(start + idx * step, arr.shape[0] - 1)
        sampled.append(arr[frame_idx])
    return np.stack(sampled, axis=0).astype(np.float32)


def _motion_anchor_orientation_10frame_step5(
    motion: SonicPlannerMotion,
    *,
    frame: int,
    base_quat_wxyz: Sequence[float],
) -> np.ndarray:
    root_quats = _sample_future_frames(motion.root_quat, frame=frame, num_frames=10, step=5, dim=4)
    base_inv = _quat_conjugate(_normalize_quat(np.asarray(base_quat_wxyz, dtype=np.float32)))
    return np.stack([_quat_to_6d(_quat_mul(base_inv, _normalize_quat(quat))) for quat in root_quats], axis=0).astype(np.float32)


def _quat_to_6d(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalize_quat(np.asarray(quat_wxyz, dtype=np.float32))
    mat = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return mat[:, :2].reshape(6)


def _quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q_vec = quat_wxyz[1:4]
    w = float(quat_wxyz[0])
    uv = np.cross(q_vec, vec)
    uuv = np.cross(q_vec, uv)
    return (vec + 2.0 * (w * uv + uuv)).astype(np.float32)


def _quat_conjugate(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).copy()
    quat[1:4] *= -1.0
    return quat


def _quat_mul(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = _normalize_quat(np.asarray(a_wxyz, dtype=np.float32))
    bw, bx, by, bz = _normalize_quat(np.asarray(b_wxyz, dtype=np.float32))
    return np.asarray(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float32,
    )


def _normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-6:
        return np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.asarray((quat[3], quat[0], quat[1], quat[2]), dtype=np.float32)


def _pad_frames(values: np.ndarray, frames: int, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, dim)
    if arr.shape[0] >= frames:
        return arr[:frames]
    if arr.shape[0] == 0:
        return np.zeros((frames, dim), dtype=np.float32)
    pad = np.repeat(arr[-1:, :], frames - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)
