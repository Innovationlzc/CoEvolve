"""Dual-G1 MuJoCo runtime for official GEAR-SONIC deploy backends.

This module keeps the official single-robot SONIC deploy process intact.  It
only replaces the Python MuJoCo side so one scene can host two namespaced G1
models, each connected to its own Unitree DDS topic prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

from praxis_coop.sonic.upper_body import SONIC_DEFAULT_UPPER_BODY, UPPER_BODY_JOINT_NAMES


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
SONIC_DEFAULT_UPPER_BODY_BY_NAME = dict(zip(UPPER_BODY_JOINT_NAMES, SONIC_DEFAULT_UPPER_BODY))


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
    hide_from_policy: bool = True
    world_level: bool = False
    level_gain: float = 1.0
    level_limit_rad: float = 0.45
    world_level_phase_keyword: str = "carry_place_crouch"


@dataclass(frozen=True)
class PlaceCrouchMotorLock:
    """Stationary lower-body place posture layered over SONIC idle balance."""

    enabled: bool = True
    phase_keyword: str = "carry_place_crouch"
    standing_height_m: float = 0.80
    max_height_drop_m: float = 0.16
    max_hip_pitch_delta_rad: float = -0.26
    max_knee_delta_rad: float = 0.52
    max_ankle_pitch_delta_rad: float = -0.28
    kp: float = 40.0
    kd: float = 7.0
    torque_blend: float = 0.55
    torque_limit_fraction: float = 0.45


@dataclass
class RobotRuntimeSlots:
    spec: DualRobotSpec
    bridge: UnitreeSdk2Bridge
    root_joint_id: int
    root_qpos_adr: int
    root_dof_adr: int
    pelvis_body_id: int
    torso_body_id: int
    left_elbow_body_id: int
    left_wrist_body_id: int
    right_elbow_body_id: int
    right_wrist_body_id: int
    body_joint_ids: np.ndarray
    left_hand_joint_ids: np.ndarray
    right_hand_joint_ids: np.ndarray
    body_actuator_ids: np.ndarray
    left_hand_actuator_ids: np.ndarray
    right_hand_actuator_ids: np.ndarray
    upper_body_lock_joint_ids: np.ndarray
    upper_body_lock_joint_names: tuple[str, ...]
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
        planner_heartbeat_drop_phase_keyword: str = "",
        stop_on_fall: bool = False,
        max_sim_time_s: float | None = None,
        summary_file: Path | None = None,
        runtime_state_file: Path | None = None,
        runtime_state_write_interval_s: float = 0.1,
        place_crouch_motor_lock: PlaceCrouchMotorLock | None = PlaceCrouchMotorLock(),
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
        self.planner_heartbeat_drop_phase_keyword = str(planner_heartbeat_drop_phase_keyword)
        self.stop_on_fall = stop_on_fall
        self.max_sim_time_s = None if max_sim_time_s is None else max(0.0, float(max_sim_time_s))
        self.summary_file = Path(summary_file) if summary_file is not None else None
        self.runtime_state_file = Path(runtime_state_file) if runtime_state_file is not None else None
        self.runtime_state_write_interval_s = max(0.02, float(runtime_state_write_interval_s))
        self.place_crouch_motor_lock = place_crouch_motor_lock
        self._lowcmd_first_seen_time: dict[str, float | None] = {}
        self.fall_events: list[dict[str, object]] = []
        self._final_status = "running"
        self._last_fall_warning_time = -1.0e9
        self._last_drop_wait_log_time = -1.0e9
        self._last_lower_body_report_time = -1.0e9
        self._last_runtime_state_write_time = -1.0e9
        self._planner_phase_cache_mtime_ns: int | None = None
        self._planner_phase_cache_value = ""
        self._planner_payload_cache_mtime_ns: int | None = None
        self._planner_payload_cache_value: dict[str, Any] = {}
        self._planner_step_payload: dict[str, Any] | None = None
        self._place_crouch_phase = ""
        self._place_crouch_phase_start_time = 0.0
        self._last_place_crouch_debug: dict[str, dict[str, float | str]] = {}
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
        self._initial_root_pose = {slot.spec.robot_id: self._root_pose_report(slot) for slot in self.slots}
        self._runtime_extrema = {
            slot.spec.robot_id: {
                "min_pelvis_height_m": float("inf"),
                "max_left_knee_rad": float("-inf"),
                "max_right_knee_rad": float("-inf"),
                "max_mean_knee_rad": float("-inf"),
                "max_abs_torso_pitch_rad": 0.0,
                "max_abs_left_forearm_dz_m": 0.0,
                "max_abs_right_forearm_dz_m": 0.0,
            }
            for slot in self.slots
        }
        self.torque_limit = np.asarray(self.robot.MOTOR_EFFORT_LIMIT_LIST, dtype=np.float64)
        if len(self.torque_limit) != self.robot.NUM_JOINTS + 2 * self.robot.NUM_HAND_JOINTS:
            raise ValueError(
                "Unexpected motor effort limits length: "
                f"{len(self.torque_limit)} for body={self.robot.NUM_JOINTS} "
                f"hands={self.robot.NUM_HAND_JOINTS}"
            )
        for slot in self.slots:
            print(
                f"[praxis-dual-sonic] initial lower-body qpos {slot.spec.robot_id}: "
                f"{self._lower_body_qpos_report(slot)}",
                flush=True,
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
                f"enabled_before_drop={self.upper_body_pose_lock.enabled_before_drop} "
                f"hide_from_policy={self.upper_body_pose_lock.hide_from_policy} "
                f"world_level={self.upper_body_pose_lock.world_level} "
                f"level_gain={self.upper_body_pose_lock.level_gain:.2f} "
                f"level_limit_rad={self.upper_body_pose_lock.level_limit_rad:.2f} "
                f"world_level_phase_keyword={self.upper_body_pose_lock.world_level_phase_keyword}",
                flush=True,
            )
            for slot in self.slots:
                mapping = self._upper_body_lock_mapping_report(slot)
                print(
                    f"[praxis-dual-sonic] upper-body lock mapping {slot.spec.robot_id}: "
                    + ", ".join(mapping),
                    flush=True,
                )
                leg_like = [name for name in mapping if any(token in name for token in ("hip", "knee", "ankle"))]
                if leg_like:
                    print(
                        f"[praxis-dual-sonic] WARNING upper-body lock includes lower-body actuator(s): {leg_like}",
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
                f"planner_heartbeat_max_age_s={self.planner_heartbeat_max_age_s:.2f} "
                f"planner_heartbeat_drop_phase_keyword={self.planner_heartbeat_drop_phase_keyword or '<any>'}",
                flush=True,
            )
        if self.place_crouch_motor_lock is not None and self.place_crouch_motor_lock.enabled:
            lock = self.place_crouch_motor_lock
            print(
                "[praxis-dual-sonic] stationary place crouch motor lock enabled "
                f"phase_keyword={lock.phase_keyword} kp={lock.kp:.1f} kd={lock.kd:.1f} "
                f"standing_height={lock.standing_height_m:.2f} "
                f"max_height_drop={lock.max_height_drop_m:.2f}",
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
            if self._final_status == "running":
                self._final_status = "stopped"
            self.close()

    def step(self) -> None:
        ctrl = np.zeros(self.model.nu, dtype=np.float64)
        self._planner_step_payload = self._read_current_planner_payload()
        try:
            self._track_lowcmd_readiness()
            self._maybe_auto_drop()
            for slot in self.slots:
                obs = self._prepare_obs(slot)
                self._mask_upper_body_lock_policy_observation(slot, obs)
                slot.bridge.PublishLowState(obs)
                self._apply_elastic_band(slot)
                self._write_robot_torques(slot, ctrl)
                self._apply_upper_body_pose_lock(slot, ctrl)
                self._apply_place_crouch_motor_lock(slot, ctrl)
        finally:
            self._planner_step_payload = None

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self._track_runtime_extrema()
        self._maybe_write_runtime_state()
        self._check_fall()
        self._maybe_report_lower_body_state()
        if self.max_sim_time_s is not None and float(self.data.time) >= self.max_sim_time_s:
            self._final_status = "max_sim_time_reached"
            self._running = False

    def _track_lowcmd_readiness(self) -> None:
        for slot in self.slots:
            try:
                with slot.bridge.low_cmd_lock:
                    low_cmd_received = bool(slot.bridge.low_cmd_received)
            except AttributeError:
                low_cmd_received = bool(getattr(slot.bridge, "low_cmd_received", False))
            if low_cmd_received and self._lowcmd_first_seen_time.get(slot.spec.robot_id) is None:
                self._lowcmd_first_seen_time[slot.spec.robot_id] = float(self.data.time)
                target_report = self._lower_body_lowcmd_report(slot)
                print(
                    f"[praxis-dual-sonic] body lowcmd first seen: {slot.spec.robot_id} "
                    f"at sim_t={self.data.time:.2f}s target_q={target_report}",
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
        payload = self._current_planner_payload()
        phase = str(payload.get("phase", ""))
        keyword = self.planner_heartbeat_drop_phase_keyword
        match_values = (
            phase,
            str(payload.get("skill_node", "")),
            str(payload.get("sync_group", "")),
            str(payload.get("motion_axis", "")),
        )
        if keyword and not any(keyword in value for value in match_values):
            return False, (
                f"planner heartbeat phase={phase or '<empty>'} does not contain "
                f"{keyword}; skill_node={payload.get('skill_node', '<empty>')} "
                f"sync_group={payload.get('sync_group', '<empty>')}"
            )
        return True, f"fresh planner heartbeat age={age:.2f}s phase={phase or '<empty>'}"

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
        if self._final_status == "running":
            self._final_status = "stopped"
        self._write_summary()
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
        left_elbow_body_id = self._body_id(f"{spec.prefix}left_elbow_link")
        left_wrist_body_id = self._body_id(f"{spec.prefix}left_wrist_roll_link")
        right_elbow_body_id = self._body_id(f"{spec.prefix}right_elbow_link")
        right_wrist_body_id = self._body_id(f"{spec.prefix}right_wrist_roll_link")
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
            upper_body_lock_joint_names,
            upper_body_lock_actuator_ids,
            upper_body_lock_targets,
            upper_body_lock_start_q,
        ) = self._upper_body_lock_slots(spec)
        elastic_band = ElasticBand()
        torso_anchor = self.data.xpos[torso_body_id].copy()
        # Keep the virtual band at the authored standing height. The upstream
        # single-G1 default uses z=1.0, but in the manual dual-G1 scene the
        # torso starts around 0.85m; pulling it to 1.0m leaves the feet
        # partially unloaded and causes a harsh drop when the band is released.
        elastic_band.point = np.array([torso_anchor[0], torso_anchor[1], torso_anchor[2]], dtype=np.float64)

        return RobotRuntimeSlots(
            spec=spec,
            bridge=bridge,
            root_joint_id=root_joint_id,
            root_qpos_adr=int(self.model.jnt_qposadr[root_joint_id]),
            root_dof_adr=int(self.model.jnt_dofadr[root_joint_id]),
            pelvis_body_id=pelvis_body_id,
            torso_body_id=torso_body_id,
            left_elbow_body_id=left_elbow_body_id,
            left_wrist_body_id=left_wrist_body_id,
            right_elbow_body_id=right_elbow_body_id,
            right_wrist_body_id=right_wrist_body_id,
            body_joint_ids=body_joint_ids_np,
            left_hand_joint_ids=left_hand_joint_ids_np,
            right_hand_joint_ids=right_hand_joint_ids_np,
            body_actuator_ids=self._actuator_ids_for_joints(body_joint_ids_np),
            left_hand_actuator_ids=self._actuator_ids_for_joints(left_hand_joint_ids_np),
            right_hand_actuator_ids=self._actuator_ids_for_joints(right_hand_joint_ids_np),
            upper_body_lock_joint_ids=upper_body_lock_joint_ids,
            upper_body_lock_joint_names=upper_body_lock_joint_names,
            upper_body_lock_actuator_ids=upper_body_lock_actuator_ids,
            upper_body_lock_targets=upper_body_lock_targets,
            upper_body_lock_start_q=upper_body_lock_start_q,
            elastic_band=elastic_band,
        )

    def _upper_body_lock_slots(
        self,
        spec: DualRobotSpec,
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
        if self.upper_body_pose_lock is None:
            empty_i = np.asarray([], dtype=np.int32)
            empty_f = np.asarray([], dtype=np.float64)
            return empty_i, (), empty_i, empty_f, empty_f
        if len(self.upper_body_pose_lock.target) != len(UPPER_BODY_JOINT_NAMES):
            raise ValueError(
                "upper_body_pose_lock target length mismatch: "
                f"{len(self.upper_body_pose_lock.target)} != {len(UPPER_BODY_JOINT_NAMES)}"
            )
        joint_ids: list[int] = []
        joint_names: list[str] = []
        targets: list[float] = []
        for name, target in zip(UPPER_BODY_JOINT_NAMES, self.upper_body_pose_lock.target):
            if name.startswith("waist") and not self.upper_body_pose_lock.include_waist:
                continue
            joint_id = self._joint_id(f"{spec.prefix}{name}_joint")
            joint_ids.append(joint_id)
            joint_names.append(name)
            targets.append(float(target))
        joint_ids_np = np.asarray(joint_ids, dtype=np.int32)
        actuator_ids_np = self._actuator_ids_for_joints(joint_ids_np)
        start_q = self.data.qpos[self.model.jnt_qposadr[joint_ids_np]].copy()
        return (
            joint_ids_np,
            tuple(joint_names),
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

    def _mask_upper_body_lock_policy_observation(self, slot: RobotRuntimeSlots, obs: dict[str, Any]) -> None:
        lock = self.upper_body_pose_lock
        if lock is None or not lock.hide_from_policy or len(slot.upper_body_lock_joint_ids) == 0:
            return
        joint_to_body_index = {int(joint_id): index for index, joint_id in enumerate(slot.body_joint_ids)}
        for joint_id in slot.upper_body_lock_joint_ids:
            joint_id_int = int(joint_id)
            body_index = joint_to_body_index.get(joint_id_int)
            if body_index is None:
                continue
            joint_name = self.model.joint(joint_id_int).name
            unprefixed = joint_name[len(slot.spec.prefix) :].removesuffix("_joint")
            default_q = SONIC_DEFAULT_UPPER_BODY_BY_NAME.get(unprefixed)
            if default_q is None:
                continue
            obs["body_q"][body_index] = float(default_q)
            obs["body_dq"][body_index] = 0.0
            obs["body_ddq"][body_index] = 0.0
            obs["body_tau_est"][body_index] = 0.0

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
        if lock.world_level:
            targets = self._world_level_upper_body_targets(slot, targets, alpha)
        q = self.data.qpos[qpos_adrs]
        dq = self.data.qvel[dof_adrs]
        torques = lock.kp * (targets - q) - lock.kd * dq
        limits = self._effective_actuator_torque_limits(slot.upper_body_lock_actuator_ids)
        torques = np.clip(torques, limits[:, 0], limits[:, 1])
        ctrl[slot.upper_body_lock_actuator_ids] = torques

    def _world_level_upper_body_targets(
        self,
        slot: RobotRuntimeSlots,
        targets: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        lock = self.upper_body_pose_lock
        if lock is None:
            return targets
        if not self._world_level_phase_active(lock):
            return targets
        torso_pitch = _pitch_from_wxyz(self.data.xquat[slot.torso_body_id])
        compensation = -float(lock.level_gain) * torso_pitch * float(alpha)
        limit = max(0.0, float(lock.level_limit_rad))
        compensation = float(np.clip(compensation, -limit, limit))
        if abs(compensation) <= 1.0e-9:
            return targets

        adjusted = targets.copy()
        for index, name in enumerate(slot.upper_body_lock_joint_names):
            if name in ("left_shoulder_pitch", "right_shoulder_pitch"):
                joint_id = int(slot.upper_body_lock_joint_ids[index])
                joint_range = self.model.jnt_range[joint_id]
                adjusted[index] = float(np.clip(targets[index] + compensation, joint_range[0], joint_range[1]))
        return adjusted

    def _world_level_phase_active(self, lock: UpperBodyPoseLock) -> bool:
        keyword = lock.world_level_phase_keyword
        if not keyword:
            return True
        phase = self._current_planner_phase()
        return keyword in phase

    def _apply_place_crouch_motor_lock(self, slot: RobotRuntimeSlots, ctrl: np.ndarray) -> None:
        lock = self.place_crouch_motor_lock
        if lock is None or not lock.enabled:
            return
        payload = self._current_planner_payload()
        phase = str(payload.get("phase", ""))
        if lock.phase_keyword not in phase:
            if self._place_crouch_phase:
                self._place_crouch_phase = ""
            self._last_place_crouch_debug.clear()
            return

        if phase != self._place_crouch_phase:
            self._place_crouch_phase = phase
            self._place_crouch_phase_start_time = float(self.data.time)

        target_height = _payload_float(payload, "height", lock.standing_height_m)
        if target_height < 0.0:
            target_height = lock.standing_height_m
        height_drop = max(0.0, lock.standing_height_m - target_height)
        crouch = min(1.0, height_drop / max(lock.max_height_drop_m, 1.0e-6))
        duration_s = max(0.1, _payload_float(payload, "duration_s", 0.1))
        elapsed_s = max(0.0, float(self.data.time) - self._place_crouch_phase_start_time)
        phase_alpha = min(1.0, elapsed_s / duration_s)
        phase_alpha = phase_alpha * phase_alpha * (3.0 - 2.0 * phase_alpha)
        if phase.endswith(".lower"):
            crouch *= phase_alpha
        elif phase.endswith(".rise"):
            crouch *= 1.0 - phase_alpha

        if crouch <= 1.0e-6:
            return

        deltas = {
            "left_hip_pitch": lock.max_hip_pitch_delta_rad * crouch,
            "right_hip_pitch": lock.max_hip_pitch_delta_rad * crouch,
            "left_knee": lock.max_knee_delta_rad * crouch,
            "right_knee": lock.max_knee_delta_rad * crouch,
            "left_ankle_pitch": lock.max_ankle_pitch_delta_rad * crouch,
            "right_ankle_pitch": lock.max_ankle_pitch_delta_rad * crouch,
        }
        for unprefixed, delta in deltas.items():
            joint_id = self._joint_id(f"{slot.spec.prefix}{unprefixed}_joint")
            actuator_id = self.actuator_by_joint[int(joint_id)]
            command_index = self._body_command_index(slot, joint_id)
            with slot.bridge.low_cmd_lock:
                lowcmd = slot.bridge.low_cmd.motor_cmd[command_index]
                base_q = float(lowcmd.q)
                base_dq = float(lowcmd.dq)
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            target = base_q + float(delta)
            q = float(self.data.qpos[qpos_adr])
            dq = float(self.data.qvel[dof_adr])
            torque = lock.kp * (target - q) + lock.kd * (base_dq - dq)
            limits = self._effective_actuator_torque_limits(np.asarray([actuator_id], dtype=np.int32))[0]
            soft_limits = np.asarray(limits, dtype=np.float64) * float(lock.torque_limit_fraction)
            torque = float(np.clip(torque, soft_limits[0], soft_limits[1]))
            blend = float(np.clip(lock.torque_blend, 0.0, 1.0))
            ctrl[actuator_id] = (1.0 - blend) * ctrl[actuator_id] + blend * torque
        self._last_place_crouch_debug[slot.spec.robot_id] = {
            "phase": phase,
            "target_height_m": float(target_height),
            "crouch": float(crouch),
            "phase_alpha": float(phase_alpha),
            "pelvis_height_m": float(self.data.xpos[slot.pelvis_body_id][2]),
            "control_mode": "lowcmd_target_plus_delta",
        }

    def _current_planner_phase(self) -> str:
        return str(self._current_planner_payload().get("phase", ""))

    def _current_planner_payload(self) -> dict[str, Any]:
        if self._planner_step_payload is not None:
            return self._planner_step_payload
        return self._read_current_planner_payload()

    def _read_current_planner_payload(self) -> dict[str, Any]:
        if self.planner_heartbeat_file is None:
            return {}
        try:
            stat = self.planner_heartbeat_file.stat()
        except OSError:
            return {}
        if stat.st_mtime_ns == self._planner_payload_cache_mtime_ns:
            return self._planner_payload_cache_value
        self._planner_payload_cache_mtime_ns = stat.st_mtime_ns
        try:
            payload = json.loads(self.planner_heartbeat_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._planner_payload_cache_value = {}
            self._planner_phase_cache_value = ""
            return {}
        if not isinstance(payload, dict):
            self._planner_payload_cache_value = {}
            self._planner_phase_cache_value = ""
            return {}
        self._planner_payload_cache_value = payload
        self._planner_phase_cache_value = str(payload.get("phase", ""))
        self._planner_phase_cache_mtime_ns = stat.st_mtime_ns
        return payload

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

    def _upper_body_lock_mapping_report(self, slot: RobotRuntimeSlots) -> list[str]:
        report = []
        for joint_id, actuator_id in zip(slot.upper_body_lock_joint_ids, slot.upper_body_lock_actuator_ids):
            joint_name = self.model.joint(int(joint_id)).name
            actuator_name = self.model.actuator(int(actuator_id)).name
            report.append(f"{joint_name}->{actuator_name}")
        return report

    def _lower_body_qpos_report(self, slot: RobotRuntimeSlots) -> str:
        names = (
            "left_hip_pitch",
            "left_knee",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_pitch",
        )
        parts = []
        for name in names:
            joint_id = self._joint_id(f"{slot.spec.prefix}{name}_joint")
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            parts.append(f"{name}={self.data.qpos[qpos_adr]:+.3f}")
        return ", ".join(parts)

    def _lower_body_qpos_dict(self, slot: RobotRuntimeSlots) -> dict[str, float]:
        names = (
            "left_hip_pitch",
            "left_knee",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_pitch",
        )
        return {name: self._joint_qpos(slot, name) for name in names}

    def _lower_body_lowcmd_report(self, slot: RobotRuntimeSlots) -> str:
        # Official deploy lowcmd uses the same MuJoCo body-joint order as
        # slot.body_joint_ids, so report the command index paired with the joint.
        wanted = {
            "left_hip_pitch_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "right_hip_pitch_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
        }
        parts = []
        with slot.bridge.low_cmd_lock:
            for command_index, joint_id in enumerate(slot.body_joint_ids):
                joint_name = self.model.joint(int(joint_id)).name
                unprefixed = joint_name[len(slot.spec.prefix) :]
                if unprefixed in wanted:
                    q = slot.bridge.low_cmd.motor_cmd[command_index].q
                    parts.append(f"{unprefixed.removesuffix('_joint')}[{command_index}]={q:+.3f}")
        return ", ".join(parts)

    def _maybe_report_lower_body_state(self, *, interval_s: float = 2.0) -> None:
        if self._elastic_band_drop_time is None:
            return
        now = float(self.data.time)
        if now - self._last_lower_body_report_time < interval_s:
            return
        self._last_lower_body_report_time = now
        for slot in self.slots:
            try:
                with slot.bridge.low_cmd_lock:
                    has_lowcmd = bool(slot.bridge.low_cmd_received)
            except AttributeError:
                has_lowcmd = bool(getattr(slot.bridge, "low_cmd_received", False))
            target = self._lower_body_lowcmd_report(slot) if has_lowcmd else "<missing>"
            print(
                f"[praxis-dual-sonic] lower-body live {slot.spec.robot_id} "
                f"sim_t={now:.2f}s qpos=({self._lower_body_qpos_report(slot)}) "
                f"target_q=({target})",
                flush=True,
            )

    def _track_runtime_extrema(self) -> None:
        for slot in self.slots:
            extrema = self._runtime_extrema[slot.spec.robot_id]
            pelvis_height = float(self.data.xpos[slot.pelvis_body_id][2])
            left_knee = self._joint_qpos(slot, "left_knee")
            right_knee = self._joint_qpos(slot, "right_knee")
            torso_pitch = _pitch_from_wxyz(self.data.xquat[slot.torso_body_id])
            left_forearm_dz = self._forearm_dz(slot, left=True)
            right_forearm_dz = self._forearm_dz(slot, left=False)
            extrema["min_pelvis_height_m"] = min(extrema["min_pelvis_height_m"], pelvis_height)
            extrema["max_left_knee_rad"] = max(extrema["max_left_knee_rad"], left_knee)
            extrema["max_right_knee_rad"] = max(extrema["max_right_knee_rad"], right_knee)
            extrema["max_mean_knee_rad"] = max(extrema["max_mean_knee_rad"], 0.5 * (left_knee + right_knee))
            extrema["max_abs_torso_pitch_rad"] = max(extrema["max_abs_torso_pitch_rad"], abs(torso_pitch))
            extrema["max_abs_left_forearm_dz_m"] = max(
                extrema["max_abs_left_forearm_dz_m"], abs(left_forearm_dz)
            )
            extrema["max_abs_right_forearm_dz_m"] = max(
                extrema["max_abs_right_forearm_dz_m"], abs(right_forearm_dz)
            )

    def _joint_qpos(self, slot: RobotRuntimeSlots, unprefixed_joint_name: str) -> float:
        joint_id = self._joint_id(f"{slot.spec.prefix}{unprefixed_joint_name}_joint")
        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        return float(self.data.qpos[qpos_adr])

    def _forearm_dz(self, slot: RobotRuntimeSlots, *, left: bool) -> float:
        if left:
            elbow = self.data.xpos[slot.left_elbow_body_id]
            wrist = self.data.xpos[slot.left_wrist_body_id]
        else:
            elbow = self.data.xpos[slot.right_elbow_body_id]
            wrist = self.data.xpos[slot.right_wrist_body_id]
        return float(wrist[2] - elbow[2])

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
                self.fall_events.append({"sim_t": now, "fallen": fallen})
                print(f"Warning: robot fallen at sim_t={now:.2f}s ({', '.join(fallen)})", flush=True)
                self._last_fall_warning_time = now
            if self.reset_on_fall:
                mujoco.mj_resetData(self.model, self.data)
            if self.stop_on_fall:
                self._final_status = "robot_fallen"
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

    def _write_summary(self) -> None:
        if self.summary_file is None:
            return
        import json

        self.summary_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": self._final_status,
            "sim_time_s": float(self.data.time),
            "fall_event_count": len(self.fall_events),
            "fall_events": self.fall_events[-20:],
            "elastic_band_dropped": self._elastic_band_drop_time is not None,
            "upper_body_lock_enabled": self.upper_body_pose_lock is not None,
            "max_sim_time_s": self.max_sim_time_s,
            "robots": {
                slot.spec.robot_id: {
                    "initial_root_pose": self._initial_root_pose[slot.spec.robot_id],
                    "root_pose": self._root_pose_report(slot),
                    "root_displacement": self._root_displacement_report(slot),
                    "pelvis_height_m": float(self.data.xpos[slot.pelvis_body_id][2]),
                    "extrema": self._finite_extrema(slot.spec.robot_id),
                    "lower_body_qpos": self._lower_body_qpos_report(slot),
                }
                for slot in self.slots
            },
        }
        self.summary_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _maybe_write_runtime_state(self) -> None:
        if self.runtime_state_file is None:
            return
        now = float(self.data.time)
        if now - self._last_runtime_state_write_time < self.runtime_state_write_interval_s:
            return
        self._write_runtime_state()
        self._last_runtime_state_write_time = now

    def _write_runtime_state(self) -> None:
        if self.runtime_state_file is None:
            return
        payload = self._runtime_state_payload()
        self.runtime_state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.runtime_state_file.with_suffix(self.runtime_state_file.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self.runtime_state_file)

    def _runtime_state_payload(self) -> dict[str, object]:
        robot_payload = {}
        yaws = []
        for slot in self.slots:
            pose = self._root_pose_report(slot)
            yaw = float(pose["yaw_rad"])
            yaws.append(yaw)
            robot_payload[slot.spec.robot_id] = {
                **pose,
                "yaw_deg": float(np.degrees(yaw)),
                "pelvis_height_m": float(self.data.xpos[slot.pelvis_body_id][2]),
                "lower_body_qpos": self._lower_body_qpos_dict(slot),
                "place_crouch": self._last_place_crouch_debug.get(slot.spec.robot_id, {}),
            }
        mean_yaw = _circular_mean(yaws)
        return {
            "source": "mujoco_live_root_pose",
            "time_unix": time.time(),
            "sim_time_s": float(self.data.time),
            "elastic_band_dropped": self._elastic_band_drop_time is not None,
            "mean_heading_yaw_rad": mean_yaw,
            "mean_heading_yaw_deg": float(np.degrees(mean_yaw)),
            "convention": "+yaw=counterclockwise, -yaw=clockwise, zero=world +X",
            "robots": robot_payload,
        }

    def _finite_extrema(self, robot_id: str) -> dict[str, float]:
        extrema = self._runtime_extrema[robot_id]
        return {
            key: float(value) if np.isfinite(value) else 0.0
            for key, value in extrema.items()
        }

    def _root_pose_report(self, slot: RobotRuntimeSlots) -> dict[str, float]:
        root = slot.root_qpos_adr
        qpos = self.data.qpos[root : root + 7]
        return {
            "x": float(qpos[0]),
            "y": float(qpos[1]),
            "z": float(qpos[2]),
            "yaw_rad": _yaw_from_wxyz(qpos[3:7]),
        }

    def _root_displacement_report(self, slot: RobotRuntimeSlots) -> dict[str, float]:
        initial = self._initial_root_pose[slot.spec.robot_id]
        current = self._root_pose_report(slot)
        dx = current["x"] - initial["x"]
        dy = current["y"] - initial["y"]
        return {
            "dx": dx,
            "dy": dy,
            "distance_xy": float(np.hypot(dx, dy)),
            "dyaw_rad": current["yaw_rad"] - initial["yaw_rad"],
        }

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

    def _body_command_index(self, slot: RobotRuntimeSlots, joint_id: int) -> int:
        matches = np.flatnonzero(slot.body_joint_ids == int(joint_id))
        if len(matches) != 1:
            joint_name = self.model.joint(int(joint_id)).name
            raise ValueError(f"Could not map body joint to lowcmd index: {joint_name}")
        return int(matches[0])

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


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _circular_mean(angles: list[float]) -> float:
    if not angles:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _pitch_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat)
    sin_pitch = 2.0 * (w * y - z * x)
    return float(np.arcsin(np.clip(sin_pitch, -1.0, 1.0)))


def _payload_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
