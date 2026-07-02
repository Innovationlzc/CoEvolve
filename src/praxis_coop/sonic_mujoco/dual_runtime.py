"""Dual-G1 MuJoCo runtime for official GEAR-SONIC deploy backends.

This module keeps the official single-robot SONIC deploy process intact.  It
only replaces the Python MuJoCo side so one scene can host two namespaced G1
models, each connected to its own Unitree DDS topic prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

from praxis_coop.sonic.upper_body import UPPER_BODY_JOINT_NAMES


GEAR_SONIC_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "GR00T-WholeBodyControl"


BODY_JOINT_MARKERS = (
    "hip",
    "knee",
    "ankle",
    "waist",
    "shoulder",
    "elbow",
    "wrist",
)


@dataclass(frozen=True)
class DualRobotSpec:
    robot_id: str
    prefix: str
    topic_prefix: str


@dataclass(frozen=True)
class UpperBodyPoseLock:
    """Motor-side symmetric arm pose lock layered on top of SONIC locomotion."""

    target: tuple[float, ...]
    kp: float = 18.0
    kd: float = 1.2
    ramp_s: float = 8.0
    include_waist: bool = False
    start_after_drop_s: float = 0.0
    enabled_before_drop: bool = False


@dataclass
class RobotRuntimeSlots:
    spec: DualRobotSpec
    bridge: UnitreeSdk2Bridge
    root_joint_id: int
    root_qpos_adr: int
    root_dof_adr: int
    pelvis_body_id: int
    torso_body_id: int
    body_joint_ids: np.ndarray
    left_hand_joint_ids: np.ndarray
    right_hand_joint_ids: np.ndarray
    body_actuator_ids: np.ndarray
    left_hand_actuator_ids: np.ndarray
    right_hand_actuator_ids: np.ndarray
    upper_body_lock_joint_ids: np.ndarray
    upper_body_lock_actuator_ids: np.ndarray
    upper_body_lock_targets: np.ndarray
    upper_body_lock_start_q: np.ndarray
    elastic_band: ElasticBand


class DualG1SonicMujocoRuntime:
    """Run two official SONIC backends against one namespaced MuJoCo scene."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        robot_specs: tuple[DualRobotSpec, DualRobotSpec] = (
            DualRobotSpec("front", "front_", "front"),
            DualRobotSpec("back", "back_", "back"),
        ),
        onscreen: bool = True,
        reset_on_fall: bool = False,
        upper_body_pose_lock: UpperBodyPoseLock | None = None,
        require_lowcmd_before_drop: bool = True,
        lowcmd_warmup_s: float = 2.0,
        auto_drop_when_ready: bool = False,
        min_drop_time_s: float = 0.0,
        planner_heartbeat_file: Path | None = None,
        planner_heartbeat_max_age_s: float = 2.0,
        stop_on_fall: bool = False,
    ) -> None:
        self.config = dict(config)
        self.robot_specs = robot_specs
        self.onscreen = onscreen
        self.reset_on_fall = reset_on_fall
        self.upper_body_pose_lock = upper_body_pose_lock
        self._elastic_band_drop_time: float | None = None
        self.require_lowcmd_before_drop = require_lowcmd_before_drop
        self.lowcmd_warmup_s = max(0.0, float(lowcmd_warmup_s))
        self.auto_drop_when_ready = auto_drop_when_ready
        self.min_drop_time_s = max(0.0, float(min_drop_time_s))
        self.planner_heartbeat_file = Path(planner_heartbeat_file) if planner_heartbeat_file is not None else None
        self.planner_heartbeat_max_age_s = max(0.0, float(planner_heartbeat_max_age_s))
        self.stop_on_fall = stop_on_fall
        self._lowcmd_first_seen_time: dict[str, float | None] = {}
        self._last_fall_warning_time = -1.0e9
        self._last_drop_wait_log_time = -1.0e9
        self.robot = Robot(self.config)
        self.sim_dt = float(self.config["SIMULATE_DT"])
        self.viewer_dt = float(self.config.get("VIEWER_DT", 0.02))
        self._running = True

        xml_path = GEAR_SONIC_ROOT / self.config["ROBOT_SCENE"]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt
        mujoco.mj_forward(self.model, self.data)

        self.actuator_by_joint = self._build_actuator_by_joint()
        self.initialize_channels()
        self.slots = tuple(self._make_slots(spec) for spec in self.robot_specs)
        self._lowcmd_first_seen_time = {slot.spec.robot_id: None for slot in self.slots}
        self.torque_limit = np.asarray(self.robot.MOTOR_EFFORT_LIMIT_LIST, dtype=np.float64)
        if len(self.torque_limit) != self.robot.NUM_JOINTS + 2 * self.robot.NUM_HAND_JOINTS:
            raise ValueError(
                "Unexpected motor effort limits length: "
                f"{len(self.torque_limit)} for body={self.robot.NUM_JOINTS} "
                f"hands={self.robot.NUM_HAND_JOINTS}"
            )

        self.viewer = None
        if self.onscreen:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self._key_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.azimuth = -80
            self.viewer.cam.elevation = -25
            self.viewer.cam.distance = 5.5
            self.viewer.cam.lookat = self._scene_lookat()
        else:
            mujoco.mj_forward(self.model, self.data)
        if self.upper_body_pose_lock is not None:
            print(
                "[praxis-dual-sonic] motor-side upper-body pose lock enabled "
                f"joints={len(self.slots[0].upper_body_lock_joint_ids)} "
                f"kp={self.upper_body_pose_lock.kp:.2f} kd={self.upper_body_pose_lock.kd:.2f} "
                f"ramp_s={self.upper_body_pose_lock.ramp_s:.2f} "
                f"include_waist={self.upper_body_pose_lock.include_waist} "
                f"start_after_drop_s={self.upper_body_pose_lock.start_after_drop_s:.2f} "
                f"enabled_before_drop={self.upper_body_pose_lock.enabled_before_drop}",
                flush=True,
            )
        if self.config.get("ENABLE_ELASTIC_BAND", True):
            print(
                "[praxis-dual-sonic] guarded drop enabled "
                f"require_lowcmd={self.require_lowcmd_before_drop} "
                f"lowcmd_warmup_s={self.lowcmd_warmup_s:.2f} "
                f"auto_drop_when_ready={self.auto_drop_when_ready} "
                f"min_drop_time_s={self.min_drop_time_s:.2f} "
                f"planner_heartbeat={self.planner_heartbeat_file or '<disabled>'} "
                f"planner_heartbeat_max_age_s={self.planner_heartbeat_max_age_s:.2f}",
                flush=True,
            )

    def initialize_channels(self) -> None:
        try:
            if self.config.get("INTERFACE"):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as exc:
            print(f"Note: Channel factory initialization attempt: {exc}", flush=True)

    def start(self) -> None:
        sim_count = 0
        print(
            "[praxis-dual-sonic] runtime started; wait for deploy CONTROL and live planner commands, "
            "then press 9 in the MuJoCo window to drop both robots",
            flush=True,
        )
        try:
            while self._running and (
                (self.viewer is not None and self.viewer.is_running()) or self.viewer is None
            ):
                step_start = time.monotonic()
                self.step()
                if sim_count % max(1, int(self.viewer_dt / self.sim_dt)) == 0:
                    if self.viewer is not None:
                        self.viewer.sync()
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                sim_count += 1
        except KeyboardInterrupt:
            print("[praxis-dual-sonic] interrupted by user", flush=True)
        finally:
            self.close()

    def step(self) -> None:
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self._track_lowcmd_readiness()
        self._maybe_auto_drop()
        for slot in self.slots:
            obs = self._prepare_obs(slot)
            slot.bridge.PublishLowState(obs)
            self._apply_elastic_band(slot)
            self._write_robot_torques(slot, ctrl)
            self._apply_upper_body_pose_lock(slot, ctrl)

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self._check_fall()

    def _track_lowcmd_readiness(self) -> None:
        for slot in self.slots:
            try:
                with slot.bridge.low_cmd_lock:
                    low_cmd_received = bool(slot.bridge.low_cmd_received)
            except AttributeError:
                low_cmd_received = bool(getattr(slot.bridge, "low_cmd_received", False))
            if low_cmd_received and self._lowcmd_first_seen_time.get(slot.spec.robot_id) is None:
                self._lowcmd_first_seen_time[slot.spec.robot_id] = float(self.data.time)
                print(
                    f"[praxis-dual-sonic] body lowcmd first seen: {slot.spec.robot_id} "
                    f"at sim_t={self.data.time:.2f}s",
                    flush=True,
                )

    def _lowcmd_ready_report(self) -> tuple[bool, str]:
        missing = []
        waiting = []
        now = float(self.data.time)
        for slot in self.slots:
            first_seen = self._lowcmd_first_seen_time.get(slot.spec.robot_id)
            if first_seen is None:
                missing.append(slot.spec.robot_id)
            else:
                age = now - first_seen
                if age < self.lowcmd_warmup_s:
                    waiting.append(f"{slot.spec.robot_id}:{age:.2f}/{self.lowcmd_warmup_s:.2f}s")
        if missing:
            return False, "missing body lowcmd for " + ", ".join(missing)
        if waiting:
            return False, "warming body lowcmd " + ", ".join(waiting)
        return True, "body lowcmd ready for all robots"

    def _planner_heartbeat_ready_report(self) -> tuple[bool, str]:
        if self.planner_heartbeat_file is None:
            return True, "planner heartbeat disabled"
        try:
            stat = self.planner_heartbeat_file.stat()
        except FileNotFoundError:
            return False, f"missing planner heartbeat {self.planner_heartbeat_file}"
        age = max(0.0, time.time() - stat.st_mtime)
        if age > self.planner_heartbeat_max_age_s:
            return False, (
                f"stale planner heartbeat age={age:.2f}s "
                f"> {self.planner_heartbeat_max_age_s:.2f}s"
            )
        return True, f"fresh planner heartbeat age={age:.2f}s"

    def _maybe_auto_drop(self) -> None:
        if not self.auto_drop_when_ready or not self.config.get("ENABLE_ELASTIC_BAND", True):
            return
        if self._elastic_band_drop_time is not None:
            return
        if not any(slot.elastic_band.enable for slot in self.slots):
            return
        ready, reason = self._lowcmd_ready_report()
        if not ready:
            self._log_auto_drop_wait(reason)
            return
        if float(self.data.time) < self.min_drop_time_s:
            self._log_auto_drop_wait(
                f"sim_t={self.data.time:.2f}s < min_drop_time_s={self.min_drop_time_s:.2f}s"
            )
            return
        planner_ready, planner_reason = self._planner_heartbeat_ready_report()
        if not planner_ready:
            self._log_auto_drop_wait(planner_reason)
            return
        self._set_elastic_band(False)
        print(f"[praxis-dual-sonic] auto-drop: {reason}", flush=True)

    def _log_auto_drop_wait(self, reason: str, *, interval_s: float = 2.0) -> None:
        now = float(self.data.time)
        if now - self._last_drop_wait_log_time < interval_s:
            return
        print(f"[praxis-dual-sonic] auto-drop waiting: {reason}", flush=True)
        self._last_drop_wait_log_time = now

    def close(self) -> None:
        self._running = False
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _make_slots(self, spec: DualRobotSpec) -> RobotRuntimeSlots:
        bridge_config = dict(self.config)
        bridge_config["DDS_TOPIC_PREFIX"] = spec.topic_prefix
        bridge = UnitreeSdk2Bridge(bridge_config)

        root_joint_id = self._joint_id(f"{spec.prefix}floating_base_joint")
        pelvis_body_id = self._body_id(f"{spec.prefix}pelvis")
        torso_body_id = self._body_id(f"{spec.prefix}torso_link")
        body_joint_ids: list[int] = []
        left_hand_joint_ids: list[int] = []
        right_hand_joint_ids: list[int] = []
        for joint_id in range(self.model.njnt):
            name = self.model.joint(joint_id).name
            if not name.startswith(spec.prefix):
                continue
            unprefixed = name[len(spec.prefix) :]
            if any(marker in unprefixed for marker in BODY_JOINT_MARKERS):
                body_joint_ids.append(joint_id)
            elif "left_hand" in unprefixed:
                left_hand_joint_ids.append(joint_id)
            elif "right_hand" in unprefixed:
                right_hand_joint_ids.append(joint_id)

        if len(body_joint_ids) != self.robot.NUM_JOINTS:
            raise ValueError(f"{spec.robot_id}: expected {self.robot.NUM_JOINTS} body joints, got {len(body_joint_ids)}")
        if len(left_hand_joint_ids) != self.robot.NUM_HAND_JOINTS:
            raise ValueError(
                f"{spec.robot_id}: expected {self.robot.NUM_HAND_JOINTS} left hand joints, "
                f"got {len(left_hand_joint_ids)}"
            )
        if len(right_hand_joint_ids) != self.robot.NUM_HAND_JOINTS:
            raise ValueError(
                f"{spec.robot_id}: expected {self.robot.NUM_HAND_JOINTS} right hand joints, "
                f"got {len(right_hand_joint_ids)}"
            )

        body_joint_ids_np = np.asarray(body_joint_ids, dtype=np.int32)
        left_hand_joint_ids_np = np.asarray(left_hand_joint_ids, dtype=np.int32)
        right_hand_joint_ids_np = np.asarray(right_hand_joint_ids, dtype=np.int32)
        (
            upper_body_lock_joint_ids,
            upper_body_lock_actuator_ids,
            upper_body_lock_targets,
            upper_body_lock_start_q,
        ) = self._upper_body_lock_slots(spec)
        elastic_band = ElasticBand()
        torso_anchor = self.data.xpos[torso_body_id].copy()
        elastic_band.point = np.array([torso_anchor[0], torso_anchor[1], 1.0], dtype=np.float64)

        return RobotRuntimeSlots(
            spec=spec,
            bridge=bridge,
            root_joint_id=root_joint_id,
            root_qpos_adr=int(self.model.jnt_qposadr[root_joint_id]),
            root_dof_adr=int(self.model.jnt_dofadr[root_joint_id]),
            pelvis_body_id=pelvis_body_id,
            torso_body_id=torso_body_id,
            body_joint_ids=body_joint_ids_np,
            left_hand_joint_ids=left_hand_joint_ids_np,
            right_hand_joint_ids=right_hand_joint_ids_np,
            body_actuator_ids=self._actuator_ids_for_joints(body_joint_ids_np),
            left_hand_actuator_ids=self._actuator_ids_for_joints(left_hand_joint_ids_np),
            right_hand_actuator_ids=self._actuator_ids_for_joints(right_hand_joint_ids_np),
            upper_body_lock_joint_ids=upper_body_lock_joint_ids,
            upper_body_lock_actuator_ids=upper_body_lock_actuator_ids,
            upper_body_lock_targets=upper_body_lock_targets,
            upper_body_lock_start_q=upper_body_lock_start_q,
            elastic_band=elastic_band,
        )

    def _upper_body_lock_slots(
        self,
        spec: DualRobotSpec,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.upper_body_pose_lock is None:
            empty_i = np.asarray([], dtype=np.int32)
            empty_f = np.asarray([], dtype=np.float64)
            return empty_i, empty_i, empty_f, empty_f
        if len(self.upper_body_pose_lock.target) != len(UPPER_BODY_JOINT_NAMES):
            raise ValueError(
                "upper_body_pose_lock target length mismatch: "
                f"{len(self.upper_body_pose_lock.target)} != {len(UPPER_BODY_JOINT_NAMES)}"
            )
        joint_ids: list[int] = []
        targets: list[float] = []
        for name, target in zip(UPPER_BODY_JOINT_NAMES, self.upper_body_pose_lock.target):
            if name.startswith("waist") and not self.upper_body_pose_lock.include_waist:
                continue
            joint_id = self._joint_id(f"{spec.prefix}{name}_joint")
            joint_ids.append(joint_id)
            targets.append(float(target))
        joint_ids_np = np.asarray(joint_ids, dtype=np.int32)
        actuator_ids_np = self._actuator_ids_for_joints(joint_ids_np)
        start_q = self.data.qpos[self.model.jnt_qposadr[joint_ids_np]].copy()
        return (
            joint_ids_np,
            actuator_ids_np,
            np.asarray(targets, dtype=np.float64),
            start_q.astype(np.float64, copy=False),
        )

    def _prepare_obs(self, slot: RobotRuntimeSlots) -> dict[str, Any]:
        root_qpos = slot.root_qpos_adr
        root_dof = slot.root_dof_adr
        body_qpos_adrs = self.model.jnt_qposadr[slot.body_joint_ids]
        body_dof_adrs = self.model.jnt_dofadr[slot.body_joint_ids]
        left_qpos_adrs = self.model.jnt_qposadr[slot.left_hand_joint_ids]
        left_dof_adrs = self.model.jnt_dofadr[slot.left_hand_joint_ids]
        right_qpos_adrs = self.model.jnt_qposadr[slot.right_hand_joint_ids]
        right_dof_adrs = self.model.jnt_dofadr[slot.right_hand_joint_ids]

        secondary_vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            slot.torso_body_id,
            secondary_vel,
            1,
        )
        secondary_vel[:3], secondary_vel[3:6] = secondary_vel[3:6].copy(), secondary_vel[:3].copy()

        return {
            "floating_base_pose": self.data.qpos[root_qpos : root_qpos + 7].copy(),
            "floating_base_vel": self.data.qvel[root_dof : root_dof + 6].copy(),
            "floating_base_acc": self.data.qacc[root_dof : root_dof + 6].copy(),
            "secondary_imu_quat": self.data.xquat[slot.torso_body_id].copy(),
            "secondary_imu_vel": secondary_vel,
            "body_q": self.data.qpos[body_qpos_adrs].copy(),
            "body_dq": self.data.qvel[body_dof_adrs].copy(),
            "body_ddq": self.data.qacc[body_dof_adrs].copy(),
            "body_tau_est": self.data.actuator_force[slot.body_actuator_ids].copy(),
            "left_hand_q": self.data.qpos[left_qpos_adrs].copy(),
            "left_hand_dq": self.data.qvel[left_dof_adrs].copy(),
            "left_hand_ddq": self.data.qacc[left_dof_adrs].copy(),
            "left_hand_tau_est": self.data.actuator_force[slot.left_hand_actuator_ids].copy(),
            "right_hand_q": self.data.qpos[right_qpos_adrs].copy(),
            "right_hand_dq": self.data.qvel[right_dof_adrs].copy(),
            "right_hand_ddq": self.data.qacc[right_dof_adrs].copy(),
            "right_hand_tau_est": self.data.actuator_force[slot.right_hand_actuator_ids].copy(),
            "time": self.data.time,
        }

    def _write_robot_torques(self, slot: RobotRuntimeSlots, ctrl: np.ndarray) -> None:
        body_torques = np.zeros(self.robot.NUM_JOINTS, dtype=np.float64)
        left_hand_torques = np.zeros(self.robot.NUM_HAND_JOINTS, dtype=np.float64)
        right_hand_torques = np.zeros(self.robot.NUM_HAND_JOINTS, dtype=np.float64)
        bridge = slot.bridge

        body_qpos_adrs = self.model.jnt_qposadr[slot.body_joint_ids]
        body_dof_adrs = self.model.jnt_dofadr[slot.body_joint_ids]
        for i in range(self.robot.NUM_JOINTS):
            cmd = bridge.low_cmd.motor_cmd[i]
            body_torques[i] = (
                cmd.tau
                + cmd.kp * (cmd.q - self.data.qpos[body_qpos_adrs[i]])
                + cmd.kd * (cmd.dq - self.data.qvel[body_dof_adrs[i]])
            )

        left_qpos_adrs = self.model.jnt_qposadr[slot.left_hand_joint_ids]
        left_dof_adrs = self.model.jnt_dofadr[slot.left_hand_joint_ids]
        right_qpos_adrs = self.model.jnt_qposadr[slot.right_hand_joint_ids]
        right_dof_adrs = self.model.jnt_dofadr[slot.right_hand_joint_ids]
        for i in range(self.robot.NUM_HAND_JOINTS):
            left_cmd = bridge.left_hand_cmd.motor_cmd[i]
            left_hand_torques[i] = (
                left_cmd.tau
                + left_cmd.kp * (left_cmd.q - self.data.qpos[left_qpos_adrs[i]])
                + left_cmd.kd * (left_cmd.dq - self.data.qvel[left_dof_adrs[i]])
            )
            right_cmd = bridge.right_hand_cmd.motor_cmd[i]
            right_hand_torques[i] = (
                right_cmd.tau
                + right_cmd.kp * (right_cmd.q - self.data.qpos[right_qpos_adrs[i]])
                + right_cmd.kd * (right_cmd.dq - self.data.qvel[right_dof_adrs[i]])
            )

        all_torques = np.concatenate((body_torques, left_hand_torques, right_hand_torques))
        all_torques = np.clip(all_torques, -self.torque_limit, self.torque_limit)
        ctrl[slot.body_actuator_ids] = all_torques[: self.robot.NUM_JOINTS]
        hand_start = self.robot.NUM_JOINTS
        hand_end = hand_start + self.robot.NUM_HAND_JOINTS
        ctrl[slot.left_hand_actuator_ids] = all_torques[hand_start:hand_end]
        ctrl[slot.right_hand_actuator_ids] = all_torques[hand_end:]

    def _apply_upper_body_pose_lock(self, slot: RobotRuntimeSlots, ctrl: np.ndarray) -> None:
        lock = self.upper_body_pose_lock
        if lock is None or len(slot.upper_body_lock_joint_ids) == 0:
            return
        if not lock.enabled_before_drop:
            if self._elastic_band_drop_time is None:
                return
            lock_t = float(self.data.time) - self._elastic_band_drop_time - float(lock.start_after_drop_s)
            if lock_t < 0.0:
                return
        else:
            lock_t = float(self.data.time)
        qpos_adrs = self.model.jnt_qposadr[slot.upper_body_lock_joint_ids]
        dof_adrs = self.model.jnt_dofadr[slot.upper_body_lock_joint_ids]
        alpha = 1.0
        if lock.ramp_s > 1.0e-9:
            alpha = min(1.0, max(0.0, lock_t / float(lock.ramp_s)))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        targets = (1.0 - alpha) * slot.upper_body_lock_start_q + alpha * slot.upper_body_lock_targets
        q = self.data.qpos[qpos_adrs]
        dq = self.data.qvel[dof_adrs]
        torques = lock.kp * (targets - q) - lock.kd * dq
        limits = self._effective_actuator_torque_limits(slot.upper_body_lock_actuator_ids)
        torques = np.clip(torques, limits[:, 0], limits[:, 1])
        ctrl[slot.upper_body_lock_actuator_ids] = torques

    def _effective_actuator_torque_limits(self, actuator_ids: np.ndarray) -> np.ndarray:
        """Return usable torque limits for MuJoCo motor actuators.

        Some G1 MJCF actuators expose torque bounds through ``ctrlrange`` while
        ``forcerange`` remains the MuJoCo default ``[0, 0]``.  Clipping a PD
        torque lock by that default silently zeros the lock, so fall back to the
        control range whenever the force range is not explicitly enabled.
        """

        force_limited = np.asarray(self.model.actuator_forcelimited[actuator_ids], dtype=bool)
        force_range = np.asarray(self.model.actuator_forcerange[actuator_ids], dtype=np.float64)
        ctrl_range = np.asarray(self.model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
        valid_force_range = np.logical_and(force_limited, force_range[:, 1] > force_range[:, 0])
        return np.where(valid_force_range[:, None], force_range, ctrl_range)

    def _apply_elastic_band(self, slot: RobotRuntimeSlots) -> None:
        if not self.config.get("ENABLE_ELASTIC_BAND", True):
            return
        if slot.elastic_band.enable:
            pose = np.concatenate(
                [
                    self.data.xpos[slot.torso_body_id],
                    self.data.xquat[slot.torso_body_id],
                    np.zeros(6, dtype=np.float64),
                ]
            )
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                slot.torso_body_id,
                pose[7:13],
                0,
            )
            pose[7:10], pose[10:13] = pose[10:13].copy(), pose[7:10].copy()
            self.data.xfrc_applied[slot.torso_body_id] = slot.elastic_band.Advance(pose)
        else:
            self.data.xfrc_applied[slot.torso_body_id] = np.zeros(6, dtype=np.float64)

    def _check_fall(self) -> None:
        fallen = []
        for slot in self.slots:
            height = float(self.data.xpos[slot.pelvis_body_id][2])
            if height < 0.2:
                fallen.append(f"{slot.spec.robot_id}:{height:.3f}m")
        if fallen:
            now = float(self.data.time)
            if now - self._last_fall_warning_time >= 0.5:
                print(f"Warning: robot fallen at sim_t={now:.2f}s ({', '.join(fallen)})", flush=True)
                self._last_fall_warning_time = now
            if self.reset_on_fall:
                mujoco.mj_resetData(self.model, self.data)
            if self.stop_on_fall:
                self._running = False

    def _key_callback(self, key: int) -> None:
        try:
            import glfw
        except ImportError:
            return
        if key == glfw.KEY_9:
            dropping = any(slot.elastic_band.enable for slot in self.slots)
            if dropping and self.require_lowcmd_before_drop:
                ready, reason = self._lowcmd_ready_report()
                if not ready:
                    print(f"[praxis-dual-sonic] drop blocked: {reason}", flush=True)
                    return
                if float(self.data.time) < self.min_drop_time_s:
                    print(
                        "[praxis-dual-sonic] drop blocked: "
                        f"sim_t={self.data.time:.2f}s < min_drop_time_s={self.min_drop_time_s:.2f}s",
                        flush=True,
                    )
                    return
                planner_ready, planner_reason = self._planner_heartbeat_ready_report()
                if not planner_ready:
                    print(f"[praxis-dual-sonic] drop blocked: {planner_reason}", flush=True)
                    return
            new_state = not dropping
            self._set_elastic_band(new_state)
            print(f"[praxis-dual-sonic] ElasticBand enable: {new_state}", flush=True)
        elif key == glfw.KEY_BACKSPACE:
            mujoco.mj_resetData(self.model, self.data)
            print("[praxis-dual-sonic] scene reset", flush=True)

    def _set_elastic_band(self, enabled: bool) -> None:
        for slot in self.slots:
            slot.elastic_band.enable = enabled
        if not enabled:
            self._elastic_band_drop_time = float(self.data.time)
            if self.upper_body_pose_lock is not None:
                print(
                    "[praxis-dual-sonic] upper-body lock timer started "
                    f"(delay={self.upper_body_pose_lock.start_after_drop_s:.2f}s, "
                    f"ramp={self.upper_body_pose_lock.ramp_s:.2f}s)",
                    flush=True,
                )
        else:
            self._elastic_band_drop_time = None

    def _scene_lookat(self) -> np.ndarray:
        positions = [self.data.xpos[slot.pelvis_body_id] for slot in self.slots]
        return np.mean(np.asarray(positions), axis=0) + np.array([0.0, 0.0, 0.25])

    def _build_actuator_by_joint(self) -> dict[int, int]:
        actuator_by_joint: dict[int, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id][0])
            if joint_id >= 0:
                actuator_by_joint[joint_id] = actuator_id
        return actuator_by_joint

    def _actuator_ids_for_joints(self, joint_ids: np.ndarray) -> np.ndarray:
        actuator_ids = []
        for joint_id in joint_ids:
            actuator_id = self.actuator_by_joint.get(int(joint_id))
            if actuator_id is None:
                joint_name = self.model.joint(int(joint_id)).name
                raise ValueError(f"Could not find actuator for joint {joint_name}")
            actuator_ids.append(actuator_id)
        return np.asarray(actuator_ids, dtype=np.int32)

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Missing joint: {name}")
        return int(joint_id)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Missing body: {name}")
        return int(body_id)
