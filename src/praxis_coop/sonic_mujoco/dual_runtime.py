"""Dual-G1 MuJoCo runtime for official GEAR-SONIC deploy backends.

This module keeps the official single-robot SONIC deploy process intact.  It
only replaces the Python MuJoCo side so one scene can host two namespaced G1
models, each connected to its own Unitree DDS topic prefix.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

from praxis_coop.sonic.upper_body import (
    DEFAULT_CARRY_READY_POSE_NAME,
    ELBOW60_PAYLOAD_RECOVERY_UPPER_BODY,
    PAYLOAD_RECOVERY_POSE_NAME,
    PAYLOAD_RELEASE_UPPER_BODY,
    SONIC_DEFAULT_UPPER_BODY,
    UPPER_BODY_JOINT_NAMES,
    is_payload_recovery_runtime_event,
    payload_release_joint_targets,
)
from praxis_coop.sonic.safety import TASK_CARRY_SAFETY
from praxis_coop.sonic.runtime_state import (
    payload_recovery_trigger,
    payload_support_center_error_m,
)
from praxis_coop.sonic_mujoco.checkpoint_store import (
    checkpoint_resync_lowcmd_status,
    load_persisted_checkpoints,
    select_mujoco_state_components,
)
from praxis_coop.sonic_mujoco.manual_layout_scene import (
    DEFAULT_PAYLOAD_REARWARD_OFFSET_M,
    payload_rearward_offset_xy,
    payload_runtime_geom_properties,
    support_surface_pitch_correction,
    support_surface_roll_correction,
)
from praxis_coop.sonic_mujoco.payload_attitude import payload_support_face_attitude


GEAR_SONIC_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "GR00T-WholeBodyControl"

GLOBAL_VIEWER_AZIMUTH_DEG = -90.0
GLOBAL_VIEWER_ELEVATION_DEG = -80.0
GLOBAL_VIEWER_DISTANCE_M = 12.5
GLOBAL_OBSERVATION_CAMERA_NAME = "vlm_global_topdown"
GLOBAL_OBSERVATION_WIDTH_PX = 1280
GLOBAL_OBSERVATION_HEIGHT_PX = 720
MAX_RUNTIME_CONTACT_FORCE_N = 1.0e6
MAX_RUNTIME_ABS_QVEL = 1.0e4
MAX_RUNTIME_ABS_QACC = 1.0e6

SAFE_CHECKPOINT_STATE_SPEC: int = (
    int(mujoco.mjtState.mjSTATE_TIME)
    | int(mujoco.mjtState.mjSTATE_PHYSICS)
    # Contact-rich payload checkpoints need a warm start consistent with the
    # restored qpos/qvel.  Reusing the pre-rollback solver acceleration can
    # make the first controlled step's sparse Hessian rank-deficient.
    | int(mujoco.mjtState.mjSTATE_WARMSTART)
    | int(mujoco.mjtState.mjSTATE_CTRL)
    # Applied generalized/body forces are part of the same integration
    # instant.  Leaving their pre-rollback values in MjData mixes two physical
    # states before the first post-restore solve.
    | int(mujoco.mjtState.mjSTATE_QFRC_APPLIED)
    | int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
    | int(mujoco.mjtState.mjSTATE_EQ_ACTIVE)
    | int(mujoco.mjtState.mjSTATE_MOCAP_POS)
    | int(mujoco.mjtState.mjSTATE_MOCAP_QUAT)
)


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
PAYLOAD_RELEASE_UPPER_BODY_BY_NAME = dict(
    zip(UPPER_BODY_JOINT_NAMES, PAYLOAD_RELEASE_UPPER_BODY)
)
PAYLOAD_RECOVERY_UPPER_BODY_BY_NAME = dict(
    zip(UPPER_BODY_JOINT_NAMES, ELBOW60_PAYLOAD_RECOVERY_UPPER_BODY)
)
# Carry support is a task-level contact concept, not a palm-proxy identity.
# The long box may transfer from a palm pad onto a hand, wrist, or forearm
# during gait while remaining physically carried. Shoulder/torso contacts are
# deliberately excluded so leaning the payload against the body is not enough.
PAYLOAD_UPPER_LIMB_SUPPORT_BODY_MARKERS = (
    "_elbow_link",
    "_wrist_roll_link",
    "_wrist_pitch_link",
    "_wrist_yaw_link",
    "_hand_",
)


def is_payload_upper_limb_support_contact(geom_name: str, body_name: str) -> bool:
    """Return whether a payload contact belongs to a hand/wrist/forearm link."""

    names = (str(geom_name), str(body_name))
    return any(
        marker in name
        for name in names
        for marker in PAYLOAD_UPPER_LIMB_SUPPORT_BODY_MARKERS
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
    pose_name: str = "custom"
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
    palm_level: bool = False
    palm_level_gain: float = 1.0
    palm_level_limit_rad: float = 0.70
    palm_level_phase_keyword: str = ""
    palm_tilt_deg: float | None = None


@dataclass(frozen=True)
class PlaceCrouchMotorLock:
    """Stationary lower-body place posture layered over SONIC idle balance.

    Official GEAR-SONIC IDLE height replans still produce a systematic body-rear
    pelvis travel of about 8-9 cm. This overlay only supplies the place-height
    knee bend; it does not teleport root XY or the payload. Treat that planar
    residual as known skill behavior rather than clamping MuJoCo qpos.
    """

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


@dataclass(frozen=True)
class PayloadHold:
    """Keep the payload fixed until the robots are ready for a contact trial."""

    joint_name: str = "long_box_freejoint"
    release_after_drop_s: float | None = None
    release_at_sim_time_s: float | None = None


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
        checkpoint_dir: Path | None = None,
        place_crouch_motor_lock: PlaceCrouchMotorLock | None = PlaceCrouchMotorLock(),
        payload_hold: PayloadHold | None = None,
        lowcmd_watchdog_timeout_s: float = 1.0,
        stop_on_control_loss: bool = True,
        viewer_camera_preset: str = "workcell",
        global_observation_frame_file: Path | None = None,
        global_observation_write_interval_s: float = 0.25,
        payload_max_support_surface_tilt_deg: float = 8.0,
        payload_rearward_offset_m: float = DEFAULT_PAYLOAD_REARWARD_OFFSET_M,
        mujoco_solver: str = "newton",
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
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.place_crouch_motor_lock = place_crouch_motor_lock
        self.payload_hold = payload_hold
        self._payload_hold_released = payload_hold is None
        self._payload_hold_qpos_adr: int | None = None
        self._payload_hold_dof_adr: int | None = None
        self._payload_hold_qpos: np.ndarray | None = None
        self.lowcmd_watchdog_timeout_s = max(0.1, float(lowcmd_watchdog_timeout_s))
        self.stop_on_control_loss = bool(stop_on_control_loss)
        self.payload_max_support_surface_tilt_deg = max(
            0.0, float(payload_max_support_surface_tilt_deg)
        )
        self.payload_rearward_offset_m = max(0.0, float(payload_rearward_offset_m))
        if viewer_camera_preset not in {"workcell", "global"}:
            raise ValueError(f"unknown viewer_camera_preset: {viewer_camera_preset!r}")
        self.viewer_camera_preset = viewer_camera_preset
        if global_observation_frame_file is None and (
            viewer_camera_preset == "global" and self.runtime_state_file is not None
        ):
            global_observation_frame_file = self.runtime_state_file.with_name(
                "vlm_global_frame.png"
            )
        self.global_observation_frame_file = (
            Path(global_observation_frame_file)
            if global_observation_frame_file is not None
            else None
        )
        self.global_observation_write_interval_s = max(
            0.05,
            float(global_observation_write_interval_s),
        )
        self._global_observation_process: subprocess.Popen[bytes] | None = None
        self._global_observation_process_log: Any | None = None
        self._global_observation_snapshot_file: Path | None = None
        self._global_observation_status_file: Path | None = None
        self._global_observation_sequence = 0
        self._global_observation_last_write_wall_time = -1.0e9
        self._global_observation_last_capture_unix = 0.0
        self._global_observation_error = ""
        self._numerical_fault_latched = False
        self._runtime_instance_id = uuid4().hex
        self._runtime_state_sequence = 0
        self._interactive_viewer_closed = False
        self._closed = False
        self._lowcmd_first_seen_time: dict[str, float | None] = {}
        self._lowcmd_last_seen_wall_time: dict[str, float | None] = {}
        self.fall_events: list[dict[str, object]] = []
        self._runtime_fault: dict[str, object] = {}
        self._final_status = "running"
        self._last_fall_warning_time = -1.0e9
        self._last_drop_wait_log_time = -1.0e9
        self._last_lower_body_report_time = -1.0e9
        self._last_runtime_state_write_time = -1.0e9
        self._last_runtime_state_write_wall_time = -1.0e9
        self._planner_phase_cache_mtime_ns: int | None = None
        self._planner_phase_cache_value = ""
        self._planner_payload_cache_mtime_ns: int | None = None
        self._planner_payload_cache_value: dict[str, Any] = {}
        self._planner_step_payload: dict[str, Any] | None = None
        self._place_crouch_phase = ""
        self._place_crouch_phase_start_time = 0.0
        self._place_crouch_latched = False
        self._place_crouch_latched_height_m = (
            float(place_crouch_motor_lock.standing_height_m)
            if place_crouch_motor_lock is not None
            else 0.80
        )
        self._last_place_crouch_debug: dict[str, dict[str, float | str | bool]] = {}
        self._last_payload_event_id = ""
        self._last_payload_wait_log_time = -1.0e9
        self._payload_release_active = False
        self._payload_release_start_time = 0.0
        self._payload_release_ramp_s = 4.0
        self._payload_release_start_targets: dict[str, np.ndarray] = {}
        self._post_release_crouched_retreat_completed = False
        self._post_release_stand_recovered = False
        self._upper_body_pose_name = (
            str(upper_body_pose_lock.pose_name)
            if upper_body_pose_lock is not None
            else "none"
        )
        self._upper_body_transition_active = False
        self._upper_body_transition_start_time = 0.0
        self._upper_body_transition_ramp_s = 3.0
        self._upper_body_transition_start_targets: dict[str, np.ndarray] = {}
        self._upper_body_transition_target_pose_name = ""
        self._last_upper_body_event_id = ""
        self._last_upper_body_wait_log_time = -1.0e9
        self._upper_body_event_status: dict[str, object] = {
            "event": "none",
            "event_id": "",
            "success": False,
            "error": "",
        }
        self._checkpoint_states: dict[str, np.ndarray] = {}
        self._checkpoint_metadata: dict[str, dict[str, Any]] = {}
        self._last_checkpoint_event_id = ""
        self._checkpoint_resync_until_wall_time: float | None = None
        self._checkpoint_resync_lowcmd_deadline_wall_time: float | None = None
        self._checkpoint_resync_waiting_for_lowcmd = False
        self._planning_physics_hold = False
        self._pending_planning_hold = False
        self._checkpoint_status: dict[str, Any] = {
            "action": "none",
            "checkpoint_id": "",
            "event_id": "",
            "success": False,
            "error": "",
        }
        self.robot = Robot(self.config)
        self.sim_dt = float(self.config["SIMULATE_DT"])
        self.viewer_dt = float(self.config.get("VIEWER_DT", 0.02))
        self._running = True

        self._xml_path = GEAR_SONIC_ROOT / self.config["ROBOT_SCENE"]
        self.model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.sim_dt
        solver_name = str(mujoco_solver).strip().lower()
        solver_by_name = {
            "newton": mujoco.mjtSolver.mjSOL_NEWTON,
            "cg": mujoco.mjtSolver.mjSOL_CG,
            "pgs": mujoco.mjtSolver.mjSOL_PGS,
        }
        if solver_name not in solver_by_name:
            raise ValueError(f"unknown MuJoCo solver: {mujoco_solver!r}")
        self.mujoco_solver = solver_name
        self.model.opt.solver = solver_by_name[solver_name]
        mujoco.mj_forward(self.model, self.data)
        self._load_persisted_checkpoints()
        self.payload_joint_id = self._optional_joint_id("long_box_freejoint")
        self.payload_geom_id = self._optional_geom_id("long_box_geom")
        self.target_geom_id = self._optional_geom_id("target_marker")
        self.payload_qpos_adr = (
            int(self.model.jnt_qposadr[self.payload_joint_id]) if self.payload_joint_id is not None else None
        )
        self.payload_dof_adr = (
            int(self.model.jnt_dofadr[self.payload_joint_id]) if self.payload_joint_id is not None else None
        )
        self.payload_support_site_id = self._optional_site_id("long_box_support_pose")
        self.payload_target_qpos = self._payload_target_qpos_from_site_or_joint()
        self.payload_active = self._payload_is_active()
        self.payload_support_plate_ids = {
            spec.robot_id: tuple(
                geom_id
                for side in ("left", "right")
                if (geom_id := self._optional_geom_id(f"{spec.prefix}{side}_carry_support_plate")) is not None
            )
            for spec in self.robot_specs
        }
        self.payload_forearm_support_plate_ids = {
            spec.robot_id: tuple(
                geom_id
                for side in ("left", "right")
                if (
                    geom_id := self._optional_geom_id(
                        f"{spec.prefix}{side}_carry_forearm_support_plate"
                    )
                )
                is not None
            )
            for spec in self.robot_specs
        }
        self.payload_support_surface_ids = {
            spec.robot_id: tuple(
                geom_id
                for side in ("left", "right")
                for suffix in ("carry_support_plate", "carry_forearm_support_plate")
                if (geom_id := self._optional_geom_id(f"{spec.prefix}{side}_{suffix}")) is not None
            )
            for spec in self.robot_specs
        }
        self._initialize_payload_hold()

        self.actuator_by_joint = self._build_actuator_by_joint()
        self.initialize_channels()
        self.slots = tuple(self._make_slots(spec) for spec in self.robot_specs)
        self._startup_upper_body_lock_targets = {
            slot.spec.robot_id: slot.upper_body_lock_targets.copy()
            for slot in self.slots
        }
        self._startup_upper_body_pose_name = self._upper_body_pose_name
        self._lowcmd_first_seen_time = {slot.spec.robot_id: None for slot in self.slots}
        self._lowcmd_last_seen_wall_time = {slot.spec.robot_id: None for slot in self.slots}
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
        self._initialize_run_metrics()
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
            self._apply_viewer_camera_preset()
            if self.viewer_camera_preset == "global":
                print(
                    "[praxis-dual-sonic] interactive viewer starts at the global view; "
                    "mouse orbit/zoom remains enabled and does not affect VLM capture",
                    flush=True,
                )
        else:
            mujoco.mj_forward(self.model, self.data)
        self._initialize_global_observation_renderer()
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
                f"world_level_phase_keyword={self.upper_body_pose_lock.world_level_phase_keyword} "
                f"palm_level={self.upper_body_pose_lock.palm_level} "
                f"palm_level_gain={self.upper_body_pose_lock.palm_level_gain:.2f} "
                f"palm_level_limit_rad={self.upper_body_pose_lock.palm_level_limit_rad:.2f} "
                f"palm_tilt_deg={self.upper_body_pose_lock.palm_tilt_deg}",
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
        print(
            "[praxis-dual-sonic] control watchdog "
            f"lowcmd_timeout_s={self.lowcmd_watchdog_timeout_s:.2f} "
            f"stop_on_control_loss={self.stop_on_control_loss} "
            f"stop_on_fall={self.stop_on_fall}",
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
        if self.payload_hold is not None:
            print(
                "[praxis-dual-sonic] payload hold enabled "
                f"joint={self.payload_hold.joint_name} "
                f"release_after_drop_s={self.payload_hold.release_after_drop_s} "
                f"release_at_sim_time_s={self.payload_hold.release_at_sim_time_s}",
                flush=True,
            )

    def _initialize_payload_hold(self) -> None:
        if self.payload_hold is None:
            return
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.payload_hold.joint_name,
        )
        if joint_id < 0:
            print(
                f"[praxis-dual-sonic] WARNING payload hold joint not found: {self.payload_hold.joint_name}",
                flush=True,
            )
            self._payload_hold_released = True
            return
        self._payload_hold_qpos_adr = int(self.model.jnt_qposadr[joint_id])
        self._payload_hold_dof_adr = int(self.model.jnt_dofadr[joint_id])
        self._payload_hold_qpos = self.data.qpos[
            self._payload_hold_qpos_adr : self._payload_hold_qpos_adr + 7
        ].copy()
        self.data.qvel[self._payload_hold_dof_adr : self._payload_hold_dof_adr + 6] = 0.0

    def _apply_payload_hold(self) -> None:
        if self.payload_hold is None or self._payload_hold_released:
            return
        if self._payload_hold_should_release():
            self._payload_hold_released = True
            print(
                f"[praxis-dual-sonic] payload hold released at sim_t={self.data.time:.2f}s",
                flush=True,
            )
            return
        if (
            self._payload_hold_qpos_adr is None
            or self._payload_hold_dof_adr is None
            or self._payload_hold_qpos is None
        ):
            return
        self.data.qpos[self._payload_hold_qpos_adr : self._payload_hold_qpos_adr + 7] = self._payload_hold_qpos
        self.data.qvel[self._payload_hold_dof_adr : self._payload_hold_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _payload_hold_should_release(self) -> bool:
        hold = self.payload_hold
        if hold is None:
            return True
        sim_time = float(self.data.time)
        if hold.release_at_sim_time_s is not None and sim_time >= hold.release_at_sim_time_s:
            return True
        if hold.release_after_drop_s is not None and self._elastic_band_drop_time is not None:
            return sim_time - self._elastic_band_drop_time >= hold.release_after_drop_s
        return False

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
            while self._running:
                self._retire_closed_interactive_viewer()
                step_start = time.monotonic()
                self.step()
                if sim_count % max(1, int(self.viewer_dt / self.sim_dt)) == 0:
                    if self.viewer is not None and not self._numerical_fault_latched:
                        numerical_fault = self._runtime_numerical_fault(
                            phase="pre_viewer_sync"
                        )
                        if numerical_fault:
                            self._latch_numerical_fault(numerical_fault)
                            self._maybe_write_runtime_state_while_paused()
                        elif self.viewer.is_running():
                            self.viewer.sync()
                        else:
                            self._retire_closed_interactive_viewer()
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
            self._check_lowcmd_watchdog()
            if not self._running:
                return
            self._apply_checkpoint_runtime_event()
            self._maybe_finish_checkpoint_resync()
            if not self._running:
                return
            if self._numerical_fault_latched:
                self._maybe_write_runtime_state_while_paused()
                return
            self._apply_upper_body_runtime_event()
            self._apply_payload_runtime_event()
            self._apply_payload_hold()
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
        if (
            self._checkpoint_resync_until_wall_time is not None
            or self._planning_physics_hold
        ):
            self._maybe_write_runtime_state_while_paused()
            return
        numerical_fault = self._runtime_numerical_fault(phase="pre_step")
        if numerical_fault:
            self._latch_numerical_fault(numerical_fault)
            self._maybe_write_runtime_state_while_paused()
            return
        if not self._advance_physics_step():
            return
        self._apply_payload_hold()
        numerical_fault = self._runtime_numerical_fault(phase="post_step")
        if numerical_fault:
            self._latch_numerical_fault(numerical_fault)
            self._maybe_write_runtime_state_while_paused()
            return
        self._track_runtime_extrema()
        self._maybe_write_global_observation_frame()
        self._maybe_write_runtime_state()
        self._check_fall()
        self._maybe_report_lower_body_state()
        if self.max_sim_time_s is not None and float(self.data.time) >= self.max_sim_time_s:
            self._final_status = "max_sim_time_reached"
            self._running = False

    def _advance_physics_step(self) -> bool:
        """Advance MuJoCo once, latching native solver failure for checkpoint restore."""

        try:
            mujoco.mj_step(self.model, self.data)
        except mujoco.FatalError as exc:
            # Native solver failures must freeze the candidate while keeping
            # Term1 and its checkpoint service alive.  A trusted restore clears
            # the latched numerical fault in ``_apply_checkpoint_restore``.
            diagnostics = self._runtime_numerical_fault(phase="mj_step_fatal")
            reasons = list(diagnostics.get("reasons", ()))
            if "mujoco_fatal_error" not in reasons:
                reasons.append("mujoco_fatal_error")
            diagnostics.update(
                {
                    "phase": "mj_step_fatal",
                    "reasons": reasons,
                    "native_exception_type": type(exc).__name__,
                    "native_exception": str(exc),
                }
            )
            self._latch_numerical_fault(diagnostics)
            self._maybe_write_runtime_state_while_paused()
            return False
        return True

    def _track_lowcmd_readiness(self) -> None:
        wall_now = time.monotonic()
        for slot in self.slots:
            try:
                with slot.bridge.low_cmd_lock:
                    low_cmd_received = bool(slot.bridge.low_cmd_received)
                    new_low_cmd = bool(getattr(slot.bridge, "new_low_cmd", False))
                    if new_low_cmd:
                        slot.bridge.new_low_cmd = False
            except AttributeError:
                low_cmd_received = bool(getattr(slot.bridge, "low_cmd_received", False))
                new_low_cmd = low_cmd_received
            if new_low_cmd:
                self._lowcmd_last_seen_wall_time[slot.spec.robot_id] = wall_now
            if low_cmd_received and self._lowcmd_first_seen_time.get(slot.spec.robot_id) is None:
                self._lowcmd_first_seen_time[slot.spec.robot_id] = float(self.data.time)
                target_report = self._lower_body_lowcmd_report(slot)
                print(
                    f"[praxis-dual-sonic] body lowcmd first seen: {slot.spec.robot_id} "
                    f"at sim_t={self.data.time:.2f}s target_q={target_report}",
                    flush=True,
                )

    def _check_lowcmd_watchdog(self) -> None:
        """Fail fast when a deploy stops refreshing motor commands.

        Reusing the last received LowCmd can look healthy for a short time and
        then slowly topple the robot.  Once the support band has been dropped,
        a stale stream is a control-plane failure, not a physical rollout that
        should be allowed to continue.
        """

        if (
            self._elastic_band_drop_time is None
            or self._runtime_fault
            or self._checkpoint_resync_until_wall_time is not None
            or self._planning_physics_hold
        ):
            return
        stale = self._stale_lowcmd_ages()
        if not stale:
            return
        self._set_lowcmd_stale_fault(stale)

    def _runtime_numerical_fault(self, *, phase: str = "runtime") -> dict[str, object]:
        arrays = {
            "qpos": self.data.qpos,
            "qvel": self.data.qvel,
            "qacc": self.data.qacc,
            "ctrl": self.data.ctrl,
            "actuator_force": self.data.actuator_force,
            "qfrc_constraint": self.data.qfrc_constraint,
        }
        nonfinite = [
            name for name, values in arrays.items() if not np.all(np.isfinite(values))
        ]
        max_abs_qvel = float(np.max(np.abs(self.data.qvel), initial=0.0))
        max_abs_qacc = float(np.max(np.abs(self.data.qacc), initial=0.0))
        max_contact_force_n = 0.0
        max_contact_index = -1
        contact_force_errors: list[str] = []
        contact_force = np.zeros(6, dtype=np.float64)
        for index in range(int(self.data.ncon)):
            try:
                mujoco.mj_contactForce(
                    self.model,
                    self.data,
                    index,
                    contact_force,
                )
            except Exception as exc:  # pragma: no cover - native binding failure
                contact_force_errors.append(f"{type(exc).__name__}: {exc}")
                continue
            force_norm = float(np.linalg.norm(contact_force))
            if force_norm > max_contact_force_n:
                max_contact_force_n = force_norm
                max_contact_index = index

        reasons = []
        if nonfinite:
            reasons.append("nonfinite:" + ",".join(nonfinite))
        if max_abs_qvel > MAX_RUNTIME_ABS_QVEL:
            reasons.append(f"qvel>{MAX_RUNTIME_ABS_QVEL:.0f}")
        if max_abs_qacc > MAX_RUNTIME_ABS_QACC:
            reasons.append(f"qacc>{MAX_RUNTIME_ABS_QACC:.0f}")
        if (
            not np.isfinite(max_contact_force_n)
            or max_contact_force_n > MAX_RUNTIME_CONTACT_FORCE_N
        ):
            reasons.append(f"contact_force>{MAX_RUNTIME_CONTACT_FORCE_N:.0f}N")
        if contact_force_errors:
            reasons.append("contact_force_unreadable")
        if not reasons:
            return {}
        return {
            "phase": phase,
            "reasons": reasons,
            "nonfinite_arrays": nonfinite,
            "max_abs_qvel": max_abs_qvel,
            "max_abs_qacc": max_abs_qacc,
            "max_contact_force_n": max_contact_force_n,
            "max_contact_index": max_contact_index,
            "contact_force_errors": contact_force_errors[:4],
        }

    def _latch_numerical_fault(self, diagnostics: dict[str, object]) -> None:
        if self._numerical_fault_latched:
            return
        self._numerical_fault_latched = True
        self._runtime_fault = {
            "type": "numerical_instability",
            "sim_time_s": float(self.data.time),
            "time_unix": time.time(),
            **diagnostics,
        }
        print(
            "[praxis-dual-sonic] NUMERICAL FAULT: physics frozen pending checkpoint "
            f"restore; reasons={diagnostics.get('reasons', [])}",
            flush=True,
        )

    def _stale_lowcmd_ages(self, *, wall_now: float | None = None) -> dict[str, float]:
        now = time.monotonic() if wall_now is None else float(wall_now)
        stale: dict[str, float] = {}
        for slot in self.slots:
            last_seen = self._lowcmd_last_seen_wall_time.get(slot.spec.robot_id)
            age = float("inf") if last_seen is None else max(0.0, now - last_seen)
            if age > self.lowcmd_watchdog_timeout_s:
                stale[slot.spec.robot_id] = age
        return stale

    def _set_lowcmd_stale_fault(self, stale: dict[str, float]) -> None:
        self._runtime_fault = {
            "type": "lowcmd_stale",
            "robots": sorted(stale),
            "lowcmd_age_s": {robot_id: age for robot_id, age in stale.items()},
            "timeout_s": self.lowcmd_watchdog_timeout_s,
            "sim_time_s": float(self.data.time),
            "time_unix": time.time(),
        }
        self._set_elastic_band(True)
        self._final_status = "control_lost"
        print(
            "[praxis-dual-sonic] CONTROL FAULT: stale LowCmd; suspended both robots "
            + ", ".join(f"{robot_id}={age:.3f}s" for robot_id, age in stale.items()),
            flush=True,
        )
        self._write_runtime_state()
        if self.stop_on_control_loss:
            self._running = False

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
        if self._closed:
            return
        self._closed = True
        self._running = False
        if self._final_status == "running":
            self._final_status = "stopped"
        # The passive Viewer has its own native thread and shares MjModel/MjData.
        # Join that thread before final contact queries or summary serialization.
        self._close_interactive_viewer()
        if not self._numerical_fault_latched:
            numerical_fault = self._runtime_numerical_fault(phase="pre_close_summary")
            if numerical_fault:
                self._latch_numerical_fault(numerical_fault)
        self._write_runtime_state()
        self._write_summary()
        process = self._global_observation_process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            self._global_observation_process = None
        if self._global_observation_process_log is not None:
            self._global_observation_process_log.close()
            self._global_observation_process_log = None

    def _retire_closed_interactive_viewer(self) -> None:
        """Detach a user-closed GUI without terminating the physics backend."""

        viewer = self.viewer
        if viewer is None or viewer.is_running():
            return
        self._wait_for_viewer_native_teardown(viewer)
        self.viewer = None
        if not self._interactive_viewer_closed:
            print(
                "[praxis-dual-sonic] interactive viewer closed; physics and fixed "
                "global observation continue headlessly",
                flush=True,
            )
        self._interactive_viewer_closed = True

    def _close_interactive_viewer(self) -> None:
        """Request native Viewer shutdown at most once and wait for teardown."""

        viewer = self.viewer
        self.viewer = None
        if viewer is None:
            return
        try:
            if viewer.is_running():
                viewer.close()
        except Exception as exc:  # pragma: no cover - native shutdown compatibility
            print(
                f"[praxis-dual-sonic] viewer shutdown warning: {type(exc).__name__}: {exc}",
                flush=True,
            )
        self._wait_for_viewer_native_teardown(viewer)

    @staticmethod
    def _wait_for_viewer_native_teardown(viewer: Any, *, timeout_s: float = 2.0) -> None:
        """Keep model/data alive until MuJoCo's passive Viewer thread is gone."""

        sim_reference = getattr(viewer, "_sim", None)
        if not callable(sim_reference):
            return
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while sim_reference() is not None and time.monotonic() < deadline:
            time.sleep(0.01)

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
        if self._payload_release_active:
            release_start = self._payload_release_start_targets.get(
                slot.spec.robot_id,
                np.asarray(
                    [
                        PAYLOAD_RELEASE_UPPER_BODY_BY_NAME[name]
                        for name in slot.upper_body_lock_joint_names
                    ],
                    dtype=np.float64,
                ),
            )
            release_progress = min(
                1.0,
                max(
                    0.0,
                    (float(self.data.time) - self._payload_release_start_time)
                    / max(self._payload_release_ramp_s, 1.0e-6),
                ),
            )
            targets = np.asarray(
                payload_release_joint_targets(
                    slot.upper_body_lock_joint_names,
                    release_start,
                    release_progress,
                ),
                dtype=np.float64,
            )
        else:
            targets = self._upper_body_transition_targets(slot, targets)
            if lock.world_level:
                targets = self._world_level_upper_body_targets(slot, targets, alpha)
            if lock.palm_level:
                targets = self._palm_level_upper_body_targets(slot, targets, alpha)
            elif lock.palm_tilt_deg is not None:
                targets = self._palm_tilt_upper_body_targets(slot, targets, alpha)
        q = self.data.qpos[qpos_adrs]
        dq = self.data.qvel[dof_adrs]
        torques = lock.kp * (targets - q) - lock.kd * dq
        limits = self._effective_actuator_torque_limits(slot.upper_body_lock_actuator_ids)
        torques = np.clip(torques, limits[:, 0], limits[:, 1])
        ctrl[slot.upper_body_lock_actuator_ids] = torques

    def _upper_body_transition_targets(
        self,
        slot: RobotRuntimeSlots,
        targets: np.ndarray,
    ) -> np.ndarray:
        if not self._upper_body_transition_active:
            return targets
        start = self._upper_body_transition_start_targets.get(slot.spec.robot_id)
        if start is None or start.shape != targets.shape:
            return targets
        elapsed = max(0.0, float(self.data.time) - self._upper_body_transition_start_time)
        alpha = min(1.0, elapsed / max(self._upper_body_transition_ramp_s, 1.0e-6))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        transitioned = (1.0 - alpha) * start + alpha * targets
        if alpha >= 1.0:
            self._upper_body_transition_active = False
            if self._upper_body_transition_target_pose_name:
                self._upper_body_pose_name = self._upper_body_transition_target_pose_name
            self._upper_body_transition_start_targets.clear()
            self._upper_body_event_status.update(
                {
                    "success": True,
                    "completed": True,
                    "target_pose_name": self._upper_body_pose_name,
                    "finished_at_sim_time_s": float(self.data.time),
                }
            )
        return transitioned

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

    def _palm_level_upper_body_targets(
        self,
        slot: RobotRuntimeSlots,
        targets: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        lock = self.upper_body_pose_lock
        if lock is None:
            return targets
        if lock.palm_level_phase_keyword:
            phase = self._current_planner_phase()
            if lock.palm_level_phase_keyword not in phase:
                return targets
        plate_ids = self.payload_support_plate_ids.get(slot.spec.robot_id, ())
        if len(plate_ids) != 2:
            return targets
        root_quat = self.data.qpos[slot.root_qpos_adr + 3 : slot.root_qpos_adr + 7]
        heading = _yaw_from_wxyz(root_quat)
        adjusted = targets.copy()
        for side, geom_id in zip(("left", "right"), plate_ids):
            rotation = np.asarray(self.data.geom_xmat[int(geom_id)], dtype=np.float64).reshape(3, 3)
            normal = rotation[:, 2].tolist()
            corrections = {
                f"{side}_wrist_pitch": support_surface_pitch_correction(normal, heading),
                f"{side}_wrist_roll": support_surface_roll_correction(normal, heading),
            }
            for joint_name, raw_correction in corrections.items():
                try:
                    index = slot.upper_body_lock_joint_names.index(joint_name)
                except ValueError:
                    continue
                correction = raw_correction * float(lock.palm_level_gain) * float(alpha)
                limit = max(0.0, float(lock.palm_level_limit_rad))
                correction = float(np.clip(correction, -limit, limit))
                joint_id = int(slot.upper_body_lock_joint_ids[index])
                joint_range = self.model.jnt_range[joint_id]
                current = float(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
                adjusted[index] = float(
                    np.clip(current + correction, joint_range[0], joint_range[1])
                )
        return adjusted

    def _palm_tilt_upper_body_targets(
        self,
        slot: RobotRuntimeSlots,
        targets: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Drive both live palm support surfaces to one world-frame tilt."""

        lock = self.upper_body_pose_lock
        if lock is None or lock.palm_tilt_deg is None:
            return targets
        plate_ids = self.payload_support_plate_ids.get(slot.spec.robot_id, ())
        if len(plate_ids) != 2:
            return targets

        desired_tilt_rad = float(
            np.radians(np.clip(float(lock.palm_tilt_deg), 0.0, 80.0))
        )
        adjusted = targets.copy()
        for side, geom_id in zip(("left", "right"), plate_ids):
            rotation = np.asarray(
                self.data.geom_xmat[int(geom_id)], dtype=np.float64
            ).reshape(3, 3)
            current_tilt_rad = float(
                np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
            )
            # In the G1 wrist convention, decreasing wrist pitch raises the
            # distal palm edge. Limit each moving target so the arm PD remains
            # smooth while converging from the live loaded posture.
            correction = -(
                desired_tilt_rad - current_tilt_rad
            ) * float(alpha)
            correction = float(np.clip(correction, -0.35, 0.35))
            joint_name = f"{side}_wrist_pitch"
            try:
                index = slot.upper_body_lock_joint_names.index(joint_name)
            except ValueError:
                continue
            joint_id = int(slot.upper_body_lock_joint_ids[index])
            joint_range = self.model.jnt_range[joint_id]
            current = float(
                self.data.qpos[int(self.model.jnt_qposadr[joint_id])]
            )
            adjusted[index] = float(
                np.clip(current + correction, joint_range[0], joint_range[1])
            )
        return adjusted

    def _apply_place_crouch_motor_lock(self, slot: RobotRuntimeSlots, ctrl: np.ndarray) -> None:
        lock = self.place_crouch_motor_lock
        if lock is None or not lock.enabled:
            return
        payload = self._current_planner_payload()
        phase = str(payload.get("phase", ""))
        recovering_stand = "carry_recover_stand" in phase
        if recovering_stand:
            self._place_crouch_latched = False
            self._place_crouch_latched_height_m = float(lock.standing_height_m)
            if self._payload_release_active:
                self._post_release_stand_recovered = True
        commanded_crouch = lock.phase_keyword in phase
        if commanded_crouch:
            requested_height = _payload_float(
                payload,
                "height",
                lock.standing_height_m,
            )
            if requested_height < 0.0:
                requested_height = lock.standing_height_m
            if requested_height < lock.standing_height_m - 1.0e-6:
                self._place_crouch_latched = True
                self._place_crouch_latched_height_m = float(requested_height)
        post_release_crouch_latched = bool(
            self._payload_release_active
            and not self._post_release_stand_recovered
        )
        if (
            post_release_crouch_latched
            and "parallel_backward.backward.brake" in phase
        ):
            self._post_release_crouched_retreat_completed = True
        pre_release_crouch_latched = bool(
            self._place_crouch_latched
            and not self._payload_release_active
            and not recovering_stand
        )
        if (
            not commanded_crouch
            and not pre_release_crouch_latched
            and not post_release_crouch_latched
        ):
            if self._place_crouch_phase:
                self._place_crouch_phase = ""
            self._last_place_crouch_debug.clear()
            return

        effective_phase = (
            phase
            if commanded_crouch
            else "carry_place_crouch.latched_hold"
        )
        if effective_phase != self._place_crouch_phase:
            self._place_crouch_phase = effective_phase
            self._place_crouch_phase_start_time = float(self.data.time)

        target_height = (
            lock.standing_height_m - lock.max_height_drop_m
            if post_release_crouch_latched and not commanded_crouch
            else (
                self._place_crouch_latched_height_m
                if pre_release_crouch_latched and not commanded_crouch
                else _payload_float(payload, "height", lock.standing_height_m)
            )
        )
        if target_height < 0.0:
            target_height = lock.standing_height_m
        height_drop = max(0.0, lock.standing_height_m - target_height)
        crouch = min(1.0, height_drop / max(lock.max_height_drop_m, 1.0e-6))
        duration_s = max(0.1, _payload_float(payload, "duration_s", 0.1))
        elapsed_s = max(0.0, float(self.data.time) - self._place_crouch_phase_start_time)
        phase_alpha = min(1.0, elapsed_s / duration_s)
        phase_alpha = phase_alpha * phase_alpha * (3.0 - 2.0 * phase_alpha)
        if effective_phase.endswith(".lower"):
            crouch *= phase_alpha
        elif effective_phase.endswith(".rise"):
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
            "phase": effective_phase,
            "target_height_m": float(target_height),
            "crouch": float(crouch),
            "phase_alpha": float(phase_alpha),
            "pelvis_height_m": float(self.data.xpos[slot.pelvis_body_id][2]),
            "control_mode": "lowcmd_target_plus_delta",
            "pre_release_latched": bool(pre_release_crouch_latched),
            "post_release_latched": bool(post_release_crouch_latched),
        }

    def _apply_upper_body_runtime_event(self) -> None:
        payload = self._current_planner_payload()
        if not is_payload_recovery_runtime_event(payload):
            return
        event_id = str(
            payload.get("upper_body_event_id", "recover_payload_inward")
        )
        if event_id == self._last_upper_body_event_id:
            return
        if self.upper_body_pose_lock is None:
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": "motor-side upper-body pose lock is disabled",
            }
            return
        if not self.payload_active or self._payload_release_active:
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": "payload is not in an active carried state",
            }
            return
        contacts = self._payload_contact_report()
        supporting = set(contacts.get("support_robots", ()))
        if bool(contacts.get("floor", False)) or not {"front", "back"}.issubset(
            supporting
        ):
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": (
                    "payload must remain off the floor with continuous front/back "
                    "upper-limb support"
                ),
            }
            if self.data.time - self._last_upper_body_wait_log_time >= 1.0:
                print(
                    "[praxis-dual-sonic] payload recovery waiting for legal dual support",
                    flush=True,
                )
                self._last_upper_body_wait_log_time = float(self.data.time)
            return

        requested_pose = str(
            payload.get("upper_body_pose_name", PAYLOAD_RECOVERY_POSE_NAME)
        )
        if requested_pose != PAYLOAD_RECOVERY_POSE_NAME:
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": f"unsupported recovery pose {requested_pose!r}",
            }
            return

        if self._upper_body_transition_active:
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": "another upper-body transition is already active",
            }
            return
        if self._upper_body_pose_name != DEFAULT_CARRY_READY_POSE_NAME:
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": (
                    "payload recovery is one-way and requires starting pose "
                    f"{DEFAULT_CARRY_READY_POSE_NAME!r}; current="
                    f"{self._upper_body_pose_name!r}"
                ),
            }
            return

        payload_pose = self._payload_pose_report()
        recovery_state = {
            "payload": payload_pose,
            "checkpoint": self._checkpoint_status,
        }
        support_error_m = payload_support_center_error_m(recovery_state)
        maximum_support_error_m = TASK_CARRY_SAFETY.max_support_center_error_m
        if support_error_m is None or support_error_m > maximum_support_error_m:
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": (
                    "payload is outside the arm-recovery support envelope: "
                    f"support_center_error={support_error_m!r}m "
                    f"maximum={maximum_support_error_m:.3f}m"
                ),
            }
            return
        threshold_m = TASK_CARRY_SAFETY.warning_payload_formation_slip_m
        recovery_trigger = payload_recovery_trigger(
            recovery_state,
            minimum_outward_offset_m=threshold_m,
        )
        if not bool(recovery_trigger["active"]):
            self._upper_body_event_status = {
                "event": "recover_payload_inward",
                "event_id": event_id,
                "success": False,
                "error": (
                    "payload has not reached either preventive arm-recovery gate: "
                    f"body_forward_offset="
                    f"{recovery_trigger['body_forward_offset_m']!r}m "
                    f"(required={threshold_m:.3f}m), "
                    f"tilt={recovery_trigger['tilt_deg']!r}deg "
                    f"(required={recovery_trigger['minimum_tilt_deg']:.1f}deg)"
                ),
            }
            return

        self._upper_body_transition_ramp_s = max(
            0.1,
            _payload_float(payload, "upper_body_ramp_s", 3.0),
        )
        self._upper_body_transition_start_time = float(self.data.time)
        self._upper_body_transition_start_targets = {}
        for slot in self.slots:
            qpos_adrs = self.model.jnt_qposadr[slot.upper_body_lock_joint_ids]
            self._upper_body_transition_start_targets[slot.spec.robot_id] = (
                self.data.qpos[qpos_adrs].copy()
            )
            for index, name in enumerate(slot.upper_body_lock_joint_names):
                if name.startswith("waist"):
                    continue
                slot.upper_body_lock_targets[index] = (
                    PAYLOAD_RECOVERY_UPPER_BODY_BY_NAME[name]
                )
        self._upper_body_transition_target_pose_name = requested_pose
        self._upper_body_transition_active = True
        self._last_upper_body_event_id = event_id
        self._upper_body_event_status = {
            "event": "recover_payload_inward",
            "event_id": event_id,
            "success": False,
            "accepted": True,
            "completed": False,
            "error": "",
            "target_pose_name": requested_pose,
            "trigger_reasons": list(recovery_trigger["reasons"]),
            "outward_offset_before_m": recovery_trigger["body_forward_offset_m"],
            "minimum_outward_offset_m": threshold_m,
            "tilt_before_deg": recovery_trigger["tilt_deg"],
            "minimum_tilt_deg": recovery_trigger["minimum_tilt_deg"],
            "ramp_s": self._upper_body_transition_ramp_s,
            "started_at_sim_time_s": float(self.data.time),
        }
        print(
            "[praxis-dual-sonic] synchronized payload inward recovery started "
            f"event_id={event_id} pose={requested_pose} "
            f"ramp_s={self._upper_body_transition_ramp_s:.2f}; waist/legs unchanged",
            flush=True,
        )

    def _apply_payload_runtime_event(self) -> None:
        if self.payload_joint_id is None or self.payload_geom_id is None or self.payload_target_qpos is None:
            return
        payload = self._current_planner_payload()
        phase = str(payload.get("phase", ""))
        event = str(payload.get("payload_event", ""))
        if event == "release_payload" or "release_payload" in phase:
            self._apply_payload_release_event(payload)
            return
        if event != "place_payload_on_arms" and "place_payload_on_arms" not in phase:
            return

        event_id = str(payload.get("payload_event_id", "place_payload_on_arms"))
        force = bool(payload.get("payload_force", False))
        if event_id == self._last_payload_event_id:
            return
        if self.payload_active and not force:
            self._last_payload_event_id = event_id
            return

        if "payload_rearward_offset_m" in payload:
            requested_offset = _payload_float(
                payload,
                "payload_rearward_offset_m",
                self.payload_rearward_offset_m,
            )
            if not np.isfinite(requested_offset) or requested_offset < 0.0:
                print(
                    "[praxis-dual-sonic] ignored invalid payload_rearward_offset_m="
                    f"{requested_offset!r}",
                    flush=True,
                )
                return
            self.payload_rearward_offset_m = float(requested_offset)

        target, wait_reasons = self._live_payload_support_target()
        if target is None:
            if self.data.time - self._last_payload_wait_log_time >= 1.0:
                print(
                    "[praxis-dual-sonic] payload placement waiting: " + "; ".join(wait_reasons),
                    flush=True,
                )
                self._last_payload_wait_log_time = float(self.data.time)
            return
        x = _payload_float(payload, "payload_x", float(target[0]))
        y = _payload_float(payload, "payload_y", float(target[1]))
        z = _payload_float(payload, "payload_z", float(target[2]))
        yaw = _payload_float(payload, "payload_yaw", _yaw_from_wxyz(target[3:7]))
        quat = _quat_wxyz_from_yaw(yaw)

        qpos_adr = int(self.payload_qpos_adr)
        dof_adr = int(self.payload_dof_adr)
        self.data.qpos[qpos_adr : qpos_adr + 7] = np.asarray([x, y, z, *quat], dtype=np.float64)
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0
        self._set_payload_active(True)
        mujoco.mj_forward(self.model, self.data)
        self.payload_active = True
        self._place_crouch_latched = False
        if self.place_crouch_motor_lock is not None:
            self._place_crouch_latched_height_m = float(
                self.place_crouch_motor_lock.standing_height_m
            )
        self._payload_release_active = False
        self._payload_release_start_targets.clear()
        self._post_release_crouched_retreat_completed = False
        self._post_release_stand_recovered = False
        self._last_payload_event_id = event_id
        print(
            "[praxis-dual-sonic] payload placed on arms "
            f"event_id={event_id} live_support_pose=({x:.3f}, {y:.3f}, {z:.3f}, yaw={yaw:.3f})",
            flush=True,
        )

    def _apply_payload_release_event(self, payload: dict[str, Any]) -> None:
        event_id = str(payload.get("payload_event_id", "release_payload"))
        if event_id == self._last_payload_event_id:
            return
        if not self.payload_active or self.upper_body_pose_lock is None:
            return
        self._payload_release_ramp_s = max(
            0.1,
            _payload_float(payload, "payload_release_ramp_s", 4.0),
        )
        self._payload_release_start_time = float(self.data.time)
        self._payload_release_start_targets = {}
        for slot in self.slots:
            qpos_adrs = self.model.jnt_qposadr[slot.upper_body_lock_joint_ids]
            self._payload_release_start_targets[slot.spec.robot_id] = self.data.qpos[
                qpos_adrs
            ].copy()
        self._payload_release_active = True
        self._post_release_crouched_retreat_completed = False
        self._post_release_stand_recovered = False
        self._last_payload_event_id = event_id
        print(
            "[praxis-dual-sonic] physical payload release started "
            f"event_id={event_id} ramp_s={self._payload_release_ramp_s:.2f}",
            flush=True,
        )

    def _apply_checkpoint_runtime_event(self) -> None:
        payload = self._current_planner_payload()
        action = str(payload.get("checkpoint_action", "")).strip().lower()
        if action not in {"save", "restore"}:
            return
        checkpoint_id = _safe_checkpoint_id(str(payload.get("checkpoint_id", "")))
        event_id = str(
            payload.get("checkpoint_event_id")
            or f"{action}:{checkpoint_id}"
        )
        if not checkpoint_id or event_id == self._last_checkpoint_event_id:
            return
        self._last_checkpoint_event_id = event_id
        try:
            if action == "save":
                if bool(payload.get("checkpoint_reset_metrics", False)):
                    self._initialize_run_metrics()
                self._save_checkpoint(checkpoint_id)
            else:
                settle_s = max(0.0, _payload_float(payload, "checkpoint_settle_s", 0.0))
                hold_planning = bool(payload.get("checkpoint_hold_planning", False))
                self._pending_planning_hold = hold_planning
                if not hold_planning:
                    self._planning_physics_hold = False
                self._restore_checkpoint(checkpoint_id, settle_s=settle_s)
            self._checkpoint_status = {
                "action": action,
                "checkpoint_id": checkpoint_id,
                "event_id": event_id,
                "success": True,
                "error": "",
                "time_unix": time.time(),
                "sim_time_s": float(self.data.time),
            }
            if action == "restore":
                self._checkpoint_status["reference_pose"] = self._checkpoint_reference_pose(
                    checkpoint_id
                )
            print(
                f"[praxis-dual-sonic] checkpoint {action} ok id={checkpoint_id} "
                f"sim_t={self.data.time:.3f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - expose control-plane failure in runtime state
            self._checkpoint_status = {
                "action": action,
                "checkpoint_id": checkpoint_id,
                "event_id": event_id,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "time_unix": time.time(),
                "sim_time_s": float(self.data.time),
            }
            print(
                f"[praxis-dual-sonic] checkpoint {action} failed id={checkpoint_id}: {exc}",
                flush=True,
            )
        self._write_runtime_state()

    def _save_checkpoint(self, checkpoint_id: str) -> None:
        state_spec = SAFE_CHECKPOINT_STATE_SPEC
        state = np.empty(mujoco.mj_stateSize(self.model, state_spec), dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, state_spec)
        metadata = {
            "checkpoint_id": checkpoint_id,
            "state_spec": int(state_spec),
            "state_size": int(state.size),
            "saved_at_unix": time.time(),
            "sim_time_s": float(self.data.time),
            "payload_active": bool(self.payload_active),
            "elastic_band_drop_time": self._elastic_band_drop_time,
            "last_payload_event_id": self._last_payload_event_id,
            "payload_release_active": self._payload_release_active,
            "payload_release_ramp_s": self._payload_release_ramp_s,
            "place_crouch_latched": self._place_crouch_latched,
            "place_crouch_latched_height_m": self._place_crouch_latched_height_m,
            "post_release_crouched_retreat_completed": (
                self._post_release_crouched_retreat_completed
            ),
            "post_release_stand_recovered": self._post_release_stand_recovered,
            "payload_rearward_offset_m": self.payload_rearward_offset_m,
            "upper_body_lock": {
                "pose_name": (
                    self._upper_body_transition_target_pose_name
                    if self._upper_body_transition_active
                    else self._upper_body_pose_name
                ),
                "targets_by_robot": {
                    slot.spec.robot_id: {
                        name: float(target)
                        for name, target in zip(
                            slot.upper_body_lock_joint_names,
                            slot.upper_body_lock_targets,
                        )
                    }
                    for slot in self.slots
                },
                "last_event_id": self._last_upper_body_event_id,
                "event_status": dict(self._upper_body_event_status),
            },
            "physical_state": self._human_checkpoint_state(),
        }
        self._checkpoint_states[checkpoint_id] = state.copy()
        self._checkpoint_metadata[checkpoint_id] = metadata
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_path = self.checkpoint_dir / f"{checkpoint_id}.npz"
            tmp_state_path = self.checkpoint_dir / f".{checkpoint_id}.npz.tmp"
            with tmp_state_path.open("wb") as handle:
                np.savez_compressed(handle, state=state, state_spec=np.asarray([int(state_spec)]))
            tmp_state_path.replace(state_path)
            metadata_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            tmp_metadata_path = self.checkpoint_dir / f".{checkpoint_id}.json.tmp"
            tmp_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            tmp_metadata_path.replace(metadata_path)

    def _load_persisted_checkpoints(self) -> None:
        persisted, rejected = load_persisted_checkpoints(self.checkpoint_dir)
        for checkpoint_id, artifact in persisted.items():
            try:
                state_spec = int(artifact.metadata["state_spec"])
                expected_size = mujoco.mj_stateSize(self.model, state_spec)
                if artifact.state.size != expected_size:
                    raise ValueError(
                        f"state size {artifact.state.size} != model state size {expected_size}"
                    )
            except (TypeError, ValueError) as exc:
                rejected = (*rejected, f"{checkpoint_id}: {exc}")
                continue
            self._checkpoint_states[checkpoint_id] = artifact.state.copy()
            self._checkpoint_metadata[checkpoint_id] = dict(artifact.metadata)
        if self._checkpoint_states:
            print(
                "[praxis-dual-sonic] loaded persisted checkpoints: "
                + ", ".join(sorted(self._checkpoint_states)),
                flush=True,
            )
        for rejection in rejected:
            print(
                f"[praxis-dual-sonic] ignored invalid checkpoint artifact: {rejection}",
                flush=True,
            )

    def _restore_checkpoint(self, checkpoint_id: str, *, settle_s: float = 0.0) -> None:
        state = self._checkpoint_states.get(checkpoint_id)
        metadata = self._checkpoint_metadata.get(checkpoint_id)
        if state is None or metadata is None:
            if self.checkpoint_dir is None:
                raise FileNotFoundError(f"checkpoint {checkpoint_id!r} is not available in memory")
            state_path = self.checkpoint_dir / f"{checkpoint_id}.npz"
            metadata_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            with np.load(state_path) as loaded:
                state = np.asarray(loaded["state"], dtype=np.float64)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._checkpoint_states[checkpoint_id] = state.copy()
            self._checkpoint_metadata[checkpoint_id] = metadata

        state_spec = int(metadata["state_spec"])
        expected_size = mujoco.mj_stateSize(self.model, state_spec)
        if state.size != expected_size:
            raise ValueError(
                f"checkpoint {checkpoint_id!r} state size {state.size} != model state size {expected_size}"
            )
        restore_spec = state_spec & SAFE_CHECKPOINT_STATE_SPEC
        required_spec = int(mujoco.mjtState.mjSTATE_QPOS) | int(
            mujoco.mjtState.mjSTATE_QVEL
        )
        if restore_spec & required_spec != required_spec:
            raise ValueError(
                f"checkpoint {checkpoint_id!r} lacks required qpos/qvel state"
            )
        restore_state = select_mujoco_state_components(
            state,
            source_spec=state_spec,
            target_spec=restore_spec,
            component_size=lambda bit: mujoco.mj_stateSize(
                self.model,
                mujoco.mjtState(bit),
            ),
        )
        viewer_lock = self.viewer.lock() if self.viewer is not None else nullcontext()
        with viewer_lock:
            mujoco.mj_setState(self.model, self.data, restore_state, restore_spec)
            self.payload_active = bool(metadata.get("payload_active", False))
            self._set_payload_active(self.payload_active)
            drop_time = metadata.get("elastic_band_drop_time")
            self._elastic_band_drop_time = None if drop_time is None else float(drop_time)
            self._last_payload_event_id = str(metadata.get("last_payload_event_id", ""))
            self._payload_release_active = bool(
                metadata.get("payload_release_active", False)
            )
            self._payload_release_ramp_s = max(
                0.1,
                float(metadata.get("payload_release_ramp_s", 4.0)),
            )
            self._payload_release_start_time = (
                float(self.data.time) - self._payload_release_ramp_s
                if self._payload_release_active
                else float(self.data.time)
            )
            self._payload_release_start_targets.clear()
            self._place_crouch_latched = bool(
                metadata.get("place_crouch_latched", False)
            )
            default_crouch_height = (
                float(self.place_crouch_motor_lock.standing_height_m)
                if self.place_crouch_motor_lock is not None
                else 0.80
            )
            restored_crouch_height = float(
                metadata.get(
                    "place_crouch_latched_height_m",
                    default_crouch_height,
                )
            )
            self._place_crouch_latched_height_m = (
                restored_crouch_height
                if np.isfinite(restored_crouch_height)
                else default_crouch_height
            )
            self._post_release_crouched_retreat_completed = bool(
                metadata.get("post_release_crouched_retreat_completed", False)
            )
            self._post_release_stand_recovered = bool(
                metadata.get("post_release_stand_recovered", False)
            )
            self._restore_upper_body_lock_metadata(metadata)
            restored_offset = metadata.get("payload_rearward_offset_m")
            if restored_offset is not None:
                restored_offset = float(restored_offset)
                if np.isfinite(restored_offset) and restored_offset >= 0.0:
                    self.payload_rearward_offset_m = restored_offset
            self._reset_runtime_extrema()
            mujoco.mj_forward(self.model, self.data)
        if self._numerical_fault_latched:
            print(
                "[praxis-dual-sonic] numerical fault cleared by trusted checkpoint "
                f"restore: {checkpoint_id}",
                flush=True,
            )
            self._numerical_fault_latched = False
            if str(self._runtime_fault.get("type", "")) == "numerical_instability":
                self._runtime_fault = {}
        self._reset_sim_time_dependent_clocks_after_restore()
        if settle_s > 0.0:
            for slot in self.slots:
                slot.elastic_band.enable = False
                self.data.xfrc_applied[slot.torso_body_id] = 0.0
            wall_now = time.monotonic()
            self._checkpoint_resync_until_wall_time = wall_now + settle_s
            self._checkpoint_resync_lowcmd_deadline_wall_time = (
                self._checkpoint_resync_until_wall_time
                + max(2.0, 3.0 * self.lowcmd_watchdog_timeout_s)
            )
            self._checkpoint_resync_waiting_for_lowcmd = False
            print(
                "[praxis-dual-sonic] checkpoint physics paused "
                f"for {settle_s:.2f}s while SONIC IDLE resynchronizes",
                flush=True,
            )
        else:
            self._checkpoint_resync_until_wall_time = None
            self._checkpoint_resync_lowcmd_deadline_wall_time = None
            self._checkpoint_resync_waiting_for_lowcmd = False
        self._metric_last_sim_time = float(self.data.time)
        self._metric_last_robot_positions = {
            slot.spec.robot_id: np.asarray(
                [self.data.qpos[slot.root_qpos_adr], self.data.qpos[slot.root_qpos_adr + 1]],
                dtype=np.float64,
            )
            for slot in self.slots
        }
        if self.payload_active and self.payload_qpos_adr is not None:
            payload = self._payload_pose_report()
            self._metric_payload_was_active = True
            self._metric_last_payload_pos = np.asarray(
                [payload["x"], payload["y"], payload["z"]], dtype=np.float64
            )
            self._metric_last_payload_tilt_deg = float(payload["tilt_deg"])
        else:
            self._metric_payload_was_active = False
            self._metric_last_payload_pos = None
            self._metric_last_payload_tilt_deg = None

    def _restore_upper_body_lock_metadata(self, metadata: dict[str, Any]) -> None:
        """Restore the command target paired with a physical MuJoCo checkpoint."""

        saved = metadata.get("upper_body_lock", {})
        saved = saved if isinstance(saved, dict) else {}
        targets_by_robot = saved.get("targets_by_robot", {})
        targets_by_robot = (
            targets_by_robot if isinstance(targets_by_robot, dict) else {}
        )
        for slot in self.slots:
            startup = self._startup_upper_body_lock_targets.get(slot.spec.robot_id)
            if startup is not None and startup.shape == slot.upper_body_lock_targets.shape:
                slot.upper_body_lock_targets[:] = startup
            robot_targets = targets_by_robot.get(slot.spec.robot_id, {})
            if not isinstance(robot_targets, dict):
                continue
            for index, name in enumerate(slot.upper_body_lock_joint_names):
                if name.startswith("waist") or name not in robot_targets:
                    continue
                value = float(robot_targets[name])
                if np.isfinite(value):
                    slot.upper_body_lock_targets[index] = value
        pose_name = str(saved.get("pose_name", "")).strip()
        self._upper_body_pose_name = pose_name or self._startup_upper_body_pose_name
        self._last_upper_body_event_id = str(saved.get("last_event_id", ""))
        event_status = saved.get("event_status", {})
        self._upper_body_event_status = (
            dict(event_status)
            if isinstance(event_status, dict) and event_status
            else {
                "event": "none",
                "event_id": "",
                "success": False,
                "error": "",
            }
        )
        self._upper_body_transition_active = False
        self._upper_body_transition_start_targets.clear()
        self._upper_body_transition_target_pose_name = ""

    def _maybe_finish_checkpoint_resync(self) -> None:
        deadline = self._checkpoint_resync_until_wall_time
        wall_now = time.monotonic()
        if deadline is None:
            return
        recovery_deadline = self._checkpoint_resync_lowcmd_deadline_wall_time
        status, stale = checkpoint_resync_lowcmd_status(
            self._lowcmd_last_seen_wall_time,
            wall_now=wall_now,
            lowcmd_timeout_s=self.lowcmd_watchdog_timeout_s,
            minimum_wait_until=deadline,
            recovery_deadline=(
                recovery_deadline if recovery_deadline is not None else deadline
            ),
        )
        if status == "minimum_wait":
            return
        if status == "waiting_lowcmd":
            if not self._checkpoint_resync_waiting_for_lowcmd:
                self._checkpoint_resync_waiting_for_lowcmd = True
                print(
                    "[praxis-dual-sonic] checkpoint physics remains paused "
                    "until both LowCmd streams are fresh: "
                    + ", ".join(
                        f"{robot_id}={age:.3f}s"
                        for robot_id, age in sorted(stale.items())
                    ),
                    flush=True,
                )
            return
        if status == "fault":
            self._checkpoint_resync_until_wall_time = None
            self._checkpoint_resync_lowcmd_deadline_wall_time = None
            self._checkpoint_resync_waiting_for_lowcmd = False
            self._set_lowcmd_stale_fault(stale)
            return
        self._checkpoint_resync_until_wall_time = None
        self._checkpoint_resync_lowcmd_deadline_wall_time = None
        self._checkpoint_resync_waiting_for_lowcmd = False
        if self._pending_planning_hold:
            self._planning_physics_hold = True
            print(
                "[praxis-dual-sonic] checkpoint physics remains paused for model planning",
                flush=True,
            )
            return
        print(
            "[praxis-dual-sonic] checkpoint physics resumed after SONIC IDLE resync "
            "with both LowCmd streams fresh",
            flush=True,
        )

    def _checkpoint_reference_pose(self, checkpoint_id: str) -> dict[str, object]:
        metadata = self._checkpoint_metadata.get(checkpoint_id, {})
        physical = metadata.get("physical_state", {})
        if not isinstance(physical, dict):
            return {}
        payload = physical.get("payload", {})
        robots = physical.get("robots", {})
        compact_robots: dict[str, object] = {}
        if isinstance(robots, dict):
            for robot_id, robot in robots.items():
                if not isinstance(robot, dict):
                    continue
                root_pose = robot.get("root_pose", {})
                if isinstance(root_pose, dict):
                    compact_robots[str(robot_id)] = dict(root_pose)
        compact_payload = dict(payload) if isinstance(payload, dict) else {}
        compact_payload["active"] = bool(metadata.get("payload_active", False))
        return {"payload": compact_payload, "robots": compact_robots}

    def _reset_sim_time_dependent_clocks_after_restore(self) -> None:
        """Reset Python-side timers after MuJoCo time moves backwards.

        MuJoCo integration state includes ``data.time``. Without resetting
        these throttling clocks, a restored runtime can stop publishing live
        state until simulation time catches up with the rejected rollout.
        """

        now = float(self.data.time)
        self._last_runtime_state_write_time = -1.0e9
        self._last_lower_body_report_time = now
        self._last_fall_warning_time = now
        self._last_drop_wait_log_time = now
        self._last_payload_wait_log_time = now
        self._place_crouch_phase = ""
        self._place_crouch_phase_start_time = now
        self._last_place_crouch_debug.clear()

    def _human_checkpoint_state(self) -> dict[str, object]:
        return {
            "payload": self._payload_pose_report(),
            "robots": {
                slot.spec.robot_id: {
                    "root_pose": self._root_pose_report(slot),
                    "pelvis_height_m": float(self.data.xpos[slot.pelvis_body_id][2]),
                    "lower_body_qpos": self._lower_body_qpos_dict(slot),
                }
                for slot in self.slots
            },
        }

    def _reset_runtime_extrema(self) -> None:
        for robot_id in self._runtime_extrema:
            self._runtime_extrema[robot_id] = {
                "min_pelvis_height_m": float("inf"),
                "max_left_knee_rad": float("-inf"),
                "max_right_knee_rad": float("-inf"),
                "max_mean_knee_rad": float("-inf"),
                "max_abs_torso_pitch_rad": 0.0,
                "max_abs_left_forearm_dz_m": 0.0,
                "max_abs_right_forearm_dz_m": 0.0,
            }

    def _live_payload_support_target(
        self,
        *,
        require_ready: bool = True,
    ) -> tuple[np.ndarray | None, tuple[str, ...]]:
        """Return a collision-free payload pose from the live upper-limb pads.

        SONIC balance recovery can shift both floating bases by several
        centimetres before the carry pose settles. A compiled world-space pose
        therefore cannot safely place a narrow box on the live palm surfaces.
        """

        reasons: list[str] = []
        if self._elastic_band_drop_time is None:
            reasons.append("robots are still on the elastic band")
        if self.upper_body_pose_lock is None:
            reasons.append("carry-ready upper-body lock is disabled")
        elif self._elastic_band_drop_time is not None:
            ready_time = (
                self._elastic_band_drop_time
                + float(self.upper_body_pose_lock.start_after_drop_s)
                + float(self.upper_body_pose_lock.ramp_s)
            )
            if self.data.time < ready_time:
                reasons.append(f"upper-body lock ramp {self.data.time - self._elastic_band_drop_time:.2f}s")

        robot_centers: list[np.ndarray] = []
        robot_yaws: list[float] = []
        plate_tops: list[float] = []
        palm_normal_tilts_deg: list[float] = []
        tilted_palms = bool(
            self.upper_body_pose_lock is not None
            and self.upper_body_pose_lock.palm_tilt_deg is not None
        )
        for slot in self.slots:
            palm_ids = self.payload_support_plate_ids.get(slot.spec.robot_id, ())
            placement_plate_ids = (
                self.payload_forearm_support_plate_ids.get(slot.spec.robot_id, ())
                if tilted_palms
                else palm_ids
            )
            if len(palm_ids) != 2:
                reasons.append(f"missing wrist support plates for {slot.spec.robot_id}")
                continue
            if len(placement_plate_ids) != 2:
                reasons.append(
                    f"missing placement support plates for {slot.spec.robot_id}"
                )
                continue
            pelvis_height = float(self.data.xpos[slot.pelvis_body_id][2])
            if pelvis_height < 0.68:
                reasons.append(f"{slot.spec.robot_id} pelvis too low ({pelvis_height:.3f}m)")
            root_velocity = self.data.qvel[slot.root_dof_adr : slot.root_dof_adr + 6]
            linear_speed = float(np.linalg.norm(root_velocity[:2]))
            angular_speed = float(np.linalg.norm(root_velocity[3:6]))
            if linear_speed > 0.12:
                reasons.append(f"{slot.spec.robot_id} is translating ({linear_speed:.3f}m/s)")
            if angular_speed > 0.45:
                reasons.append(f"{slot.spec.robot_id} is rotating ({angular_speed:.3f}rad/s)")
            if len(slot.upper_body_lock_joint_ids):
                current = np.asarray(
                    [self.data.qpos[int(self.model.jnt_qposadr[joint_id])] for joint_id in slot.upper_body_lock_joint_ids],
                    dtype=np.float64,
                )
                effective_targets = np.asarray(slot.upper_body_lock_targets, dtype=np.float64)
                if self.upper_body_pose_lock is not None:
                    if self.upper_body_pose_lock.world_level:
                        effective_targets = self._world_level_upper_body_targets(
                            slot, effective_targets, 1.0
                        )
                    if self.upper_body_pose_lock.palm_level:
                        effective_targets = self._palm_level_upper_body_targets(
                            slot, effective_targets, 1.0
                        )
                    elif self.upper_body_pose_lock.palm_tilt_deg is not None:
                        effective_targets = self._palm_tilt_upper_body_targets(
                            slot, effective_targets, 1.0
                        )
                max_error = float(np.max(np.abs(current - effective_targets)))
                if max_error > 0.30:
                    reasons.append(f"{slot.spec.robot_id} arm lock error {max_error:.3f}rad")

            # Keep the rearward-offset convention anchored at the four live
            # palms. With raised palms, only placement height comes from the
            # forearm plates; using their already-rearward XY center here
            # would apply the configured rearward offset twice and put the
            # payload COM on the inner support edge.
            palm_centers = np.asarray(
                [
                    self.data.geom_xpos[geom_id]
                    for geom_id in palm_ids
                ],
                dtype=np.float64,
            )
            robot_centers.append(np.mean(palm_centers, axis=0))
            root_quat = self.data.qpos[slot.root_qpos_adr + 3 : slot.root_qpos_adr + 7]
            robot_yaws.append(_yaw_from_wxyz(root_quat))
            for geom_id in placement_plate_ids:
                rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                half_extent_z = float(np.dot(np.abs(rotation[2]), self.model.geom_size[geom_id]))
                plate_tops.append(float(self.data.geom_xpos[geom_id][2]) + half_extent_z)
            for geom_id in palm_ids:
                rotation = np.asarray(
                    self.data.geom_xmat[geom_id], dtype=np.float64
                ).reshape(3, 3)
                palm_normal_tilts_deg.append(
                    float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
                )

        if len(robot_centers) != len(self.slots) or not plate_tops:
            return None, tuple(reasons or ("live upper-limb support geometry is unavailable",))
        if max(plate_tops) - min(plate_tops) > 0.08:
            reasons.append(
                f"placement support height mismatch {max(plate_tops) - min(plate_tops):.3f}m"
            )
        if tilted_palms:
            desired_tilt_deg = float(self.upper_body_pose_lock.palm_tilt_deg)
            max_tilt_error_deg = max(
                (
                    abs(value - desired_tilt_deg)
                    for value in palm_normal_tilts_deg
                ),
                default=float("inf"),
            )
            if max_tilt_error_deg > 8.0:
                reasons.append(
                    "palm backstop tilt error "
                    f"{max_tilt_error_deg:.2f}deg exceeds 8.00deg"
                )
        else:
            max_surface_tilt = self.payload_max_support_surface_tilt_deg
            if (
                palm_normal_tilts_deg
                and max(palm_normal_tilts_deg) > max_surface_tilt
            ):
                reasons.append(
                    "palm support surface tilt "
                    f"{max(palm_normal_tilts_deg):.2f}deg exceeds "
                    f"{max_surface_tilt:.2f}deg"
                )
        reasons.extend(_payload_support_formation_wait_reasons(robot_centers, robot_yaws))
        if reasons and require_ready:
            return None, tuple(reasons)

        front_center, back_center = robot_centers
        support_center = 0.5 * (front_center + back_center)
        support_center[:2] += payload_rearward_offset_xy(robot_yaws, self.payload_rearward_offset_m)
        support_axis = back_center[:2] - front_center[:2]
        if float(np.linalg.norm(support_axis)) < 0.20:
            return None, ("robot wrist support centers are too close",)
        yaw = float(np.arctan2(support_axis[1], support_axis[0]))
        payload_half_height = float(self.model.geom_size[int(self.payload_geom_id)][2])
        # One millimetre avoids initial penetration, then gravity settles the
        # free payload onto the palm-aligned contact surfaces.
        z = float(np.mean(plate_tops) + payload_half_height + 0.001)
        return np.asarray([support_center[0], support_center[1], z, *_quat_wxyz_from_yaw(yaw)]), ()

    def _payload_is_active(self) -> bool:
        if self.payload_geom_id is None:
            return False
        geom_id = int(self.payload_geom_id)
        return bool(
            self.model.geom_rgba[geom_id][3] > 0.5
            and self.model.geom_contype[geom_id] != 0
            and self.model.geom_conaffinity[geom_id] != 0
        )

    def _set_payload_active(self, active: bool) -> None:
        if self.payload_geom_id is None:
            return
        geom_id = int(self.payload_geom_id)
        rgba, contype, conaffinity = payload_runtime_geom_properties(active)
        self.model.geom_rgba[geom_id] = np.asarray(rgba, dtype=np.float32)
        # Deferred means visually hidden at the far staging pose, not removed
        # from physics. Keeping these masks live is required for a restored
        # deferred checkpoint to remain on the floor instead of free-falling
        # through it before the next placement event.
        self.model.geom_contype[geom_id] = contype
        self.model.geom_conaffinity[geom_id] = conaffinity

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

    def _upper_body_lock_state_report(self, slot: RobotRuntimeSlots) -> dict[str, object]:
        lock = self.upper_body_pose_lock
        if lock is None or len(slot.upper_body_lock_joint_ids) == 0:
            return {"enabled": False}
        qpos_adrs = self.model.jnt_qposadr[slot.upper_body_lock_joint_ids]
        current = np.asarray(self.data.qpos[qpos_adrs], dtype=np.float64)
        target = np.asarray(slot.upper_body_lock_targets, dtype=np.float64)
        target = self._upper_body_transition_targets(slot, target)
        if lock.world_level:
            target = self._world_level_upper_body_targets(slot, target, 1.0)
        if lock.palm_level:
            target = self._palm_level_upper_body_targets(slot, target, 1.0)
        elif lock.palm_tilt_deg is not None:
            target = self._palm_tilt_upper_body_targets(slot, target, 1.0)
        error = target - current
        return {
            "enabled": True,
            "kp": float(lock.kp),
            "kd": float(lock.kd),
            "max_abs_error_rad": float(np.max(np.abs(error))),
            "joints": {
                name: {
                    "target_rad": float(target[index]),
                    "current_rad": float(current[index]),
                    "error_rad": float(error[index]),
                }
                for index, name in enumerate(slot.upper_body_lock_joint_names)
            },
        }

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
            wrench = slot.elastic_band.Advance(pose)
            self.data.xfrc_applied[slot.torso_body_id] = wrench
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
                self._runtime_fault = {
                    "type": "robot_fall",
                    "robots": list(fallen),
                    "sim_time_s": now,
                    "time_unix": time.time(),
                }
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

    def _apply_viewer_camera_preset(self) -> None:
        if self.viewer is None:
            return
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            if self.viewer_camera_preset == "global":
                self.viewer.cam.azimuth = GLOBAL_VIEWER_AZIMUTH_DEG
                self.viewer.cam.elevation = GLOBAL_VIEWER_ELEVATION_DEG
                self.viewer.cam.distance = GLOBAL_VIEWER_DISTANCE_M
                self.viewer.cam.lookat = self._global_scene_lookat()
                return
            self.viewer.cam.azimuth = -80
            self.viewer.cam.elevation = -25
            self.viewer.cam.distance = 5.5
            self.viewer.cam.lookat = self._scene_lookat()

    def _global_scene_lookat(self) -> np.ndarray:
        """Center the authored walls, target, and initial robots in one wide view."""

        global_camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "vlm_global_topdown",
        )
        if global_camera_id >= 0:
            camera_position = np.asarray(
                self.model.cam_pos[global_camera_id],
                dtype=np.float64,
            )
            return np.asarray(
                [camera_position[0], camera_position[1], 0.45],
                dtype=np.float64,
            )
        positions = [np.asarray(self.data.xpos[slot.pelvis_body_id], dtype=np.float64) for slot in self.slots]
        for geom_id in range(self.model.ngeom):
            name = self.model.geom(geom_id).name
            if name.startswith("wall_") or name == "target_marker":
                positions.append(np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64))
        points = np.asarray(positions, dtype=np.float64)
        xy_center = 0.5 * (np.min(points[:, :2], axis=0) + np.max(points[:, :2], axis=0))
        return np.asarray([xy_center[0], xy_center[1], 0.45], dtype=np.float64)

    def _initialize_global_observation_renderer(self) -> None:
        """Start fixed rendering in a process isolated from the GLFW viewer."""

        if self.global_observation_frame_file is None:
            return
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            GLOBAL_OBSERVATION_CAMERA_NAME,
        )
        if camera_id < 0:
            self._global_observation_error = (
                f"named camera {GLOBAL_OBSERVATION_CAMERA_NAME!r} is missing"
            )
            print(
                "[praxis-dual-sonic] WARNING fixed VLM capture disabled: "
                f"{self._global_observation_error}",
                flush=True,
            )
            return
        frame_path = self.global_observation_frame_file
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        self._global_observation_snapshot_file = frame_path.with_name(
            f".{frame_path.stem}.render_state.npz"
        )
        self._global_observation_status_file = frame_path.with_name(
            f".{frame_path.stem}.render_status.json"
        )
        for stale_path in (
            frame_path,
            self._global_observation_snapshot_file,
            self._global_observation_status_file,
        ):
            stale_path.unlink(missing_ok=True)
        try:
            log_path = frame_path.with_name(f"{frame_path.stem}_renderer.log")
            self._global_observation_process_log = log_path.open("ab")
            env = os.environ.copy()
            env["MUJOCO_GL"] = "egl"
            src_root = Path(__file__).resolve().parents[2]
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                str(src_root)
                if not existing_pythonpath
                else f"{src_root}{os.pathsep}{existing_pythonpath}"
            )
            self._global_observation_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "praxis_coop.sonic_mujoco.offscreen_renderer",
                    "--model",
                    str(self._xml_path),
                    "--snapshot",
                    str(self._global_observation_snapshot_file),
                    "--output",
                    str(frame_path),
                    "--status",
                    str(self._global_observation_status_file),
                    "--camera",
                    GLOBAL_OBSERVATION_CAMERA_NAME,
                    "--width",
                    str(GLOBAL_OBSERVATION_WIDTH_PX),
                    "--height",
                    str(GLOBAL_OBSERVATION_HEIGHT_PX),
                ],
                env=env,
                stdout=self._global_observation_process_log,
                stderr=subprocess.STDOUT,
            )
            self._maybe_write_global_observation_frame(force=True)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                self._refresh_global_observation_status()
                if self._global_observation_last_capture_unix > 0.0:
                    break
                process = self._global_observation_process
                if process is None or process.poll() is not None:
                    break
                time.sleep(0.05)
            print(
                "[praxis-dual-sonic] fixed VLM rendering isolated in EGL subprocess",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface capture failure in runtime state
            process = self._global_observation_process
            if process is not None and process.poll() is None:
                process.terminate()
            self._global_observation_process = None
            self._global_observation_error = f"{type(exc).__name__}: {exc}"
            print(
                "[praxis-dual-sonic] WARNING fixed VLM capture initialization failed: "
                f"{self._global_observation_error}",
                flush=True,
            )

    def _maybe_write_global_observation_frame(self, *, force: bool = False) -> None:
        process = self._global_observation_process
        frame_path = self.global_observation_frame_file
        snapshot_path = self._global_observation_snapshot_file
        if process is None or frame_path is None or snapshot_path is None:
            return
        if process.poll() is not None:
            self._global_observation_error = (
                f"offscreen renderer exited with code {process.returncode}"
            )
            return
        wall_now = time.monotonic()
        if (
            not force
            and wall_now - self._global_observation_last_write_wall_time
            < self.global_observation_write_interval_s
        ):
            self._refresh_global_observation_status()
            return
        try:
            viewer_lock = self.viewer.lock() if self.viewer is not None else nullcontext()
            with viewer_lock:
                qpos = self.data.qpos.copy()
                geom_rgba = self.model.geom_rgba.copy()
                site_rgba = self.model.site_rgba.copy()
                sim_time_s = float(self.data.time)
            self._global_observation_sequence += 1
            temporary = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    qpos=qpos,
                    geom_rgba=geom_rgba,
                    site_rgba=site_rgba,
                    sim_time_s=np.asarray([sim_time_s], dtype=np.float64),
                    sequence=np.asarray(
                        [self._global_observation_sequence],
                        dtype=np.int64,
                    ),
                )
            temporary.replace(snapshot_path)
            self._global_observation_last_write_wall_time = wall_now
            self._refresh_global_observation_status()
        except Exception as exc:  # noqa: BLE001 - keep motor runtime alive, fail visual gate later
            self._global_observation_error = f"{type(exc).__name__}: {exc}"

    def _refresh_global_observation_status(self) -> None:
        status_path = self._global_observation_status_file
        if status_path is None or not status_path.is_file():
            return
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            captured_at = float(status.get("captured_at_unix", 0.0))
            available = bool(status.get("available", False))
            error = str(status.get("error", ""))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._global_observation_error = f"{type(exc).__name__}: {exc}"
            return
        if available and captured_at > 0.0:
            self._global_observation_last_capture_unix = captured_at
            self._global_observation_error = ""
        elif error:
            self._global_observation_error = error

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
            "elastic_band_dropped": not any(slot.elastic_band.enable for slot in self.slots),
            "upper_body_lock_enabled": self.upper_body_pose_lock is not None,
            "payload_hold_enabled": self.payload_hold is not None,
            "payload_hold_released": self._payload_hold_released,
            "runtime_fault": self._runtime_fault,
            "control_health": self._control_health_report(),
            "payload": self._payload_pose_report(),
            "target": self._target_report(),
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
        self._update_run_metrics(now)
        self._write_runtime_state()
        self._last_runtime_state_write_time = now
        self._last_runtime_state_write_wall_time = time.monotonic()

    def _maybe_write_runtime_state_while_paused(self) -> None:
        if self.runtime_state_file is None:
            return
        now = time.monotonic()
        if now - self._last_runtime_state_write_wall_time < self.runtime_state_write_interval_s:
            return
        if not self._numerical_fault_latched:
            self._maybe_write_global_observation_frame()
        self._write_runtime_state()
        self._last_runtime_state_write_wall_time = now

    def _write_runtime_state(self) -> None:
        if self.runtime_state_file is None:
            return
        self._runtime_state_sequence += 1
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
                "initial_x": float(self._initial_root_pose[slot.spec.robot_id]["x"]),
                "initial_y": float(self._initial_root_pose[slot.spec.robot_id]["y"]),
                "root_displacement": self._root_displacement_report(slot),
                "yaw_deg": float(np.degrees(yaw)),
                "pelvis_height_m": float(self.data.xpos[slot.pelvis_body_id][2]),
                "upper_body_lock": self._upper_body_lock_state_report(slot),
                "lower_body_qpos": self._lower_body_qpos_dict(slot),
                "place_crouch": self._last_place_crouch_debug.get(slot.spec.robot_id, {}),
            }
        mean_yaw = _circular_mean(yaws)
        return {
            "source": "mujoco_live_root_pose",
            "runtime_instance_id": self._runtime_instance_id,
            "state_sequence": self._runtime_state_sequence,
            "runtime_status": self._final_status,
            "runtime_fault": self._runtime_fault,
            "control_health": self._control_health_report(),
            "time_unix": time.time(),
            "sim_time_s": float(self.data.time),
            "elastic_band_dropped": not any(slot.elastic_band.enable for slot in self.slots),
            "payload_hold_enabled": self.payload_hold is not None,
            "payload_hold_released": self._payload_hold_released,
            "mujoco_solver": self.mujoco_solver,
            "mean_heading_yaw_rad": mean_yaw,
            "mean_heading_yaw_deg": float(np.degrees(mean_yaw)),
            "convention": "+yaw=counterclockwise, -yaw=clockwise, zero=world +X",
            "viewer_camera": self._viewer_camera_report(),
            "global_observation": self._global_observation_report(),
            "upper_body_pose_lock": self._upper_body_runtime_report(),
            "place_crouch_lifecycle": {
                "latched": bool(self._place_crouch_latched),
                "target_height_m": float(self._place_crouch_latched_height_m),
            },
            "post_release_lifecycle": {
                "payload_released": bool(self._payload_release_active),
                "crouch_latched": bool(
                    self._payload_release_active
                    and not self._post_release_stand_recovered
                ),
                "crouched_retreat_completed": bool(
                    self._post_release_crouched_retreat_completed
                ),
                "stand_recovered": bool(self._post_release_stand_recovered),
            },
            "checkpoint": {
                **self._checkpoint_status,
                "available_ids": sorted(self._checkpoint_states),
                "resync_active": (
                    self._checkpoint_resync_until_wall_time is not None
                ),
                "planning_physics_hold": bool(self._planning_physics_hold),
                "resync_until_sim_time_s": None,
                "resync_remaining_s": (
                    max(0.0, self._checkpoint_resync_until_wall_time - time.monotonic())
                    if self._checkpoint_resync_until_wall_time is not None
                    else 0.0
                ),
                "resync_waiting_for_fresh_lowcmd": bool(
                    self._checkpoint_resync_waiting_for_lowcmd
                ),
                "resync_mode": (
                    "physics_paused_planning"
                    if self._planning_physics_hold
                    else "physics_paused_controller_resync"
                    if self._checkpoint_resync_until_wall_time is not None
                    else "none"
                ),
            },
            "run_metrics": self._run_metrics_report(),
            "payload": self._payload_pose_report(),
            "target": self._target_report(),
            "robots": robot_payload,
        }

    def _upper_body_runtime_report(self) -> dict[str, object]:
        elapsed = max(
            0.0,
            float(self.data.time) - self._upper_body_transition_start_time,
        )
        lock = self.upper_body_pose_lock
        return {
            "configured": self.upper_body_pose_lock is not None,
            "pose_name": self._upper_body_pose_name,
            "target_pose_name": (
                self._upper_body_transition_target_pose_name
                if self._upper_body_transition_active
                else self._upper_body_pose_name
            ),
            "transition_active": bool(self._upper_body_transition_active),
            "transition_elapsed_s": (
                min(elapsed, self._upper_body_transition_ramp_s)
                if self._upper_body_transition_active
                else 0.0
            ),
            "transition_ramp_s": float(self._upper_body_transition_ramp_s),
            "kp": None if lock is None else float(lock.kp),
            "kd": None if lock is None else float(lock.kd),
            "configured_ramp_s": None if lock is None else float(lock.ramp_s),
            "palm_tilt_deg": (
                None
                if lock is None or lock.palm_tilt_deg is None
                else float(lock.palm_tilt_deg)
            ),
            "waist_and_legs_unchanged_by_runtime_events": True,
            "last_event": dict(self._upper_body_event_status),
        }

    def _target_report(self) -> dict[str, object]:
        if self.target_geom_id is None:
            return {"available": False}
        geom_id = int(self.target_geom_id)
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        top_z = float(center[2] + size[1])
        payload_half_height = (
            float(self.model.geom_size[int(self.payload_geom_id)][2])
            if self.payload_geom_id is not None
            else 0.05
        )
        return {
            "available": True,
            "geom_name": self.model.geom(geom_id).name,
            "center_x_m": float(center[0]),
            "center_y_m": float(center[1]),
            "center_z_m": float(center[2]),
            "radius_m": float(size[0]),
            "half_height_m": float(size[1]),
            "top_z_m": top_z,
            "payload_center_target_z_m": top_z + payload_half_height,
        }

    def _viewer_camera_report(self) -> dict[str, object]:
        if self.viewer is None:
            return {
                "available": False,
                "preset": self.viewer_camera_preset,
            }
        viewport = self.viewer.viewport
        return {
            "available": True,
            "preset": self.viewer_camera_preset,
            "locked_to_preset": False,
            "camera_type": "interactive_free",
            "camera_name": "",
            "azimuth_deg": float(self.viewer.cam.azimuth),
            "elevation_deg": float(self.viewer.cam.elevation),
            "distance_m": float(self.viewer.cam.distance),
            "lookat_xyz": [float(item) for item in self.viewer.cam.lookat],
            "fovy_deg": float(self.model.vis.global_.fovy),
            "viewport_width_px": None if viewport is None else int(viewport.width),
            "viewport_height_px": None if viewport is None else int(viewport.height),
        }

    def _global_observation_report(self) -> dict[str, object]:
        self._refresh_global_observation_status()
        frame_path = self.global_observation_frame_file
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            GLOBAL_OBSERVATION_CAMERA_NAME,
        )
        available = bool(
            frame_path is not None
            and camera_id >= 0
            and self._global_observation_process is not None
            and self._global_observation_process.poll() is None
            and frame_path.exists()
            and not self._global_observation_error
        )
        lookat = self._global_scene_lookat()
        camera_z = (
            float(self.data.cam_xpos[camera_id][2])
            if camera_id >= 0
            else GLOBAL_VIEWER_DISTANCE_M + float(lookat[2])
        )
        age_s = (
            max(0.0, time.time() - self._global_observation_last_capture_unix)
            if self._global_observation_last_capture_unix > 0.0
            else None
        )
        return {
            "available": available,
            "source": "mujoco_offscreen_fixed_camera",
            "camera_name": GLOBAL_OBSERVATION_CAMERA_NAME,
            "locked_to_camera": True,
            "independent_of_interactive_viewer": True,
            "renderer_backend": "egl_subprocess",
            "frame_path": "" if frame_path is None else str(frame_path),
            "captured_at_unix": self._global_observation_last_capture_unix,
            "age_s": age_s,
            "error": self._global_observation_error,
            "azimuth_deg": -90.0,
            "elevation_deg": -90.0,
            "distance_m": max(0.0, camera_z - float(lookat[2])),
            "lookat_xyz": [float(item) for item in lookat],
            "fovy_deg": (
                float(self.model.cam_fovy[camera_id])
                if camera_id >= 0
                else float(self.model.vis.global_.fovy)
            ),
            "width_px": GLOBAL_OBSERVATION_WIDTH_PX,
            "height_px": GLOBAL_OBSERVATION_HEIGHT_PX,
        }

    def _control_health_report(self) -> dict[str, object]:
        wall_now = time.monotonic()
        streams: dict[str, dict[str, object]] = {}
        for slot in self.slots:
            last_seen = self._lowcmd_last_seen_wall_time.get(slot.spec.robot_id)
            age = float("inf") if last_seen is None else max(0.0, wall_now - last_seen)
            streams[slot.spec.robot_id] = {
                "received": last_seen is not None,
                "age_s": None if not np.isfinite(age) else age,
                "fresh": bool(age <= self.lowcmd_watchdog_timeout_s),
            }
        return {
            "healthy": bool(not self._runtime_fault and all(item["fresh"] for item in streams.values())),
            "lowcmd_timeout_s": self.lowcmd_watchdog_timeout_s,
            "streams": streams,
        }

    def _initialize_run_metrics(self) -> None:
        self._metric_last_sim_time = float(self.data.time)
        self._metric_last_payload_pos: np.ndarray | None = None
        self._metric_last_payload_tilt_deg: float | None = None
        self._metric_last_robot_positions = {
            slot.spec.robot_id: np.asarray(
                [self._root_pose_report(slot)["x"], self._root_pose_report(slot)["y"]],
                dtype=np.float64,
            )
            for slot in self.slots
        }
        self._metric_payload_was_active = False
        self._metric_wall_contact_active = False
        self._metric_floor_contact_active = False
        self._run_metrics: dict[str, Any] = {
            "sample_count": 0,
            "payload_active_time_s": 0.0,
            "payload_path_length_m": 0.0,
            "robot_path_length_m": {slot.spec.robot_id: 0.0 for slot in self.slots},
            "payload_tilt_sq_deg2_s": 0.0,
            "payload_max_tilt_deg": 0.0,
            "payload_tilt_rate_abs_deg": 0.0,
            "payload_max_tilt_rate_degps": 0.0,
            "dual_support_time_s": 0.0,
            "wall_contact_count": 0,
            "wall_contact_time_s": 0.0,
            "floor_contact_count": 0,
            "floor_contact_time_s": 0.0,
            "velocity_mismatch_sq_m2ps2_s": 0.0,
            "max_velocity_mismatch_mps": 0.0,
            "idle_payload_drift_cm": 0.0,
            "max_idle_payload_speed_cmps": 0.0,
        }

    def _update_run_metrics(self, now: float) -> None:
        dt = max(0.0, float(now) - float(self._metric_last_sim_time))
        self._metric_last_sim_time = float(now)
        if dt <= 0.0:
            return
        metrics = self._run_metrics
        metrics["sample_count"] += 1

        robot_velocities = []
        for slot in self.slots:
            position = np.asarray(
                [self.data.qpos[slot.root_qpos_adr], self.data.qpos[slot.root_qpos_adr + 1]],
                dtype=np.float64,
            )
            previous = self._metric_last_robot_positions[slot.spec.robot_id]
            metrics["robot_path_length_m"][slot.spec.robot_id] += float(np.linalg.norm(position - previous))
            self._metric_last_robot_positions[slot.spec.robot_id] = position
            robot_velocities.append(
                np.asarray(self.data.qvel[slot.root_dof_adr : slot.root_dof_adr + 2], dtype=np.float64)
            )
        if len(robot_velocities) == 2:
            mismatch = float(np.linalg.norm(robot_velocities[0] - robot_velocities[1]))
            metrics["velocity_mismatch_sq_m2ps2_s"] += mismatch * mismatch * dt
            metrics["max_velocity_mismatch_mps"] = max(metrics["max_velocity_mismatch_mps"], mismatch)

        if not self.payload_active or self.payload_qpos_adr is None:
            self._metric_payload_was_active = False
            self._metric_last_payload_pos = None
            self._metric_last_payload_tilt_deg = None
            return

        payload = self._payload_pose_report()
        position = np.asarray([payload["x"], payload["y"], payload["z"]], dtype=np.float64)
        tilt_deg = float(payload["tilt_deg"])
        if self._metric_payload_was_active and self._metric_last_payload_pos is not None:
            displacement = float(np.linalg.norm(position - self._metric_last_payload_pos))
            metrics["payload_path_length_m"] += displacement
            if _planner_phase_is_idle(self._current_planner_phase()):
                metrics["idle_payload_drift_cm"] += 100.0 * displacement
        metrics["payload_active_time_s"] += dt
        metrics["payload_tilt_sq_deg2_s"] += tilt_deg * tilt_deg * dt
        metrics["payload_max_tilt_deg"] = max(metrics["payload_max_tilt_deg"], tilt_deg)
        if self._metric_last_payload_tilt_deg is not None:
            tilt_delta = abs(tilt_deg - self._metric_last_payload_tilt_deg)
            tilt_rate = tilt_delta / dt
            metrics["payload_tilt_rate_abs_deg"] += tilt_delta
            metrics["payload_max_tilt_rate_degps"] = max(
                metrics["payload_max_tilt_rate_degps"], tilt_rate
            )
        speed_cmps = 100.0 * float(payload["linear_speed_mps"])
        if _planner_phase_is_idle(self._current_planner_phase()):
            metrics["max_idle_payload_speed_cmps"] = max(
                metrics["max_idle_payload_speed_cmps"], speed_cmps
            )

        contacts = payload["contacts"]
        both_supporting = {"front", "back"}.issubset(set(contacts.get("robots", ())))
        wall_contact = bool(contacts.get("walls"))
        floor_contact = bool(contacts.get("floor"))
        if both_supporting:
            metrics["dual_support_time_s"] += dt
        if wall_contact:
            metrics["wall_contact_time_s"] += dt
            if not self._metric_wall_contact_active:
                metrics["wall_contact_count"] += 1
        if floor_contact:
            metrics["floor_contact_time_s"] += dt
            if not self._metric_floor_contact_active:
                metrics["floor_contact_count"] += 1
        self._metric_wall_contact_active = wall_contact
        self._metric_floor_contact_active = floor_contact
        self._metric_payload_was_active = True
        self._metric_last_payload_pos = position
        self._metric_last_payload_tilt_deg = tilt_deg

    def _run_metrics_report(self) -> dict[str, object]:
        metrics = self._run_metrics
        active_time = max(0.0, float(metrics["payload_active_time_s"]))
        tilt_rms = (
            float(np.sqrt(metrics["payload_tilt_sq_deg2_s"] / active_time))
            if active_time > 0.0
            else 0.0
        )
        velocity_rms = (
            float(np.sqrt(metrics["velocity_mismatch_sq_m2ps2_s"] / active_time))
            if active_time > 0.0
            else 0.0
        )
        return {
            "sample_count": int(metrics["sample_count"]),
            "payload_active_time_s": active_time,
            "payload_path_length_m": float(metrics["payload_path_length_m"]),
            "robot_path_length_m": {
                robot_id: float(distance)
                for robot_id, distance in metrics["robot_path_length_m"].items()
            },
            "payload_rms_tilt_deg": tilt_rms,
            "payload_max_tilt_deg": float(metrics["payload_max_tilt_deg"]),
            "payload_mean_abs_tilt_rate_degps": (
                float(metrics["payload_tilt_rate_abs_deg"] / active_time) if active_time > 0.0 else 0.0
            ),
            "payload_max_tilt_rate_degps": float(metrics["payload_max_tilt_rate_degps"]),
            "dual_support_time_ratio": (
                float(metrics["dual_support_time_s"] / active_time) if active_time > 0.0 else 0.0
            ),
            "wall_contact_count": int(metrics["wall_contact_count"]),
            "wall_contact_time_s": float(metrics["wall_contact_time_s"]),
            "floor_contact_count": int(metrics["floor_contact_count"]),
            "floor_contact_time_s": float(metrics["floor_contact_time_s"]),
            "robot_velocity_mismatch_rms_mps": velocity_rms,
            "robot_velocity_mismatch_max_mps": float(metrics["max_velocity_mismatch_mps"]),
            "post_stop_drift_cm": float(metrics["idle_payload_drift_cm"]),
            "post_stop_max_speed_cmps": float(metrics["max_idle_payload_speed_cmps"]),
        }

    def _finite_extrema(self, robot_id: str) -> dict[str, float]:
        extrema = self._runtime_extrema[robot_id]
        return {
            key: float(value) if np.isfinite(value) else 0.0
            for key, value in extrema.items()
        }

    def _upper_body_lock_state_dict(self, slot: RobotRuntimeSlots) -> dict[str, object]:
        lock = self.upper_body_pose_lock
        enabled_now = False
        alpha = 0.0
        if lock is not None and len(slot.upper_body_lock_joint_ids) > 0:
            if lock.enabled_before_drop:
                lock_t = float(self.data.time)
                enabled_now = lock_t >= 0.0
            elif self._elastic_band_drop_time is not None:
                lock_t = float(self.data.time) - self._elastic_band_drop_time - float(lock.start_after_drop_s)
                enabled_now = lock_t >= 0.0
            else:
                lock_t = -1.0
            if enabled_now:
                alpha = 1.0
                if lock.ramp_s > 1.0e-9:
                    alpha = min(1.0, max(0.0, lock_t / float(lock.ramp_s)))
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        qpos_adrs = self.model.jnt_qposadr[slot.upper_body_lock_joint_ids]
        current = self.data.qpos[qpos_adrs] if len(qpos_adrs) > 0 else np.asarray([], dtype=np.float64)
        target = (
            (1.0 - alpha) * slot.upper_body_lock_start_q + alpha * slot.upper_body_lock_targets
            if len(slot.upper_body_lock_joint_ids) > 0
            else np.asarray([], dtype=np.float64)
        )
        target = self._upper_body_transition_targets(slot, target)
        wanted = {
            "left_shoulder_pitch",
            "right_shoulder_pitch",
            "left_elbow",
            "right_elbow",
            "left_wrist_pitch",
            "right_wrist_pitch",
        }
        joints = {}
        for name, q, q_target in zip(slot.upper_body_lock_joint_names, current, target):
            if name.removesuffix("_joint") in wanted:
                joints[name.removesuffix("_joint")] = {
                    "q": float(q),
                    "target": float(q_target),
                    "error": float(q_target - q),
                }
        return {
            "configured": lock is not None,
            "joint_count": int(len(slot.upper_body_lock_joint_ids)),
            "enabled_before_drop": bool(lock.enabled_before_drop) if lock is not None else False,
            "enabled_now": enabled_now,
            "alpha": float(alpha),
            "pose_name": self._upper_body_pose_name,
            "target_pose_name": (
                self._upper_body_transition_target_pose_name
                if self._upper_body_transition_active
                else self._upper_body_pose_name
            ),
            "joints": joints,
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

    def _payload_pose_report(self) -> dict[str, object]:
        if self.payload_qpos_adr is None:
            return {"present": False}
        qpos = self.data.qpos[self.payload_qpos_adr : self.payload_qpos_adr + 7]
        quat = np.asarray(qpos[3:7], dtype=np.float64)
        roll, pitch, yaw = _rpy_from_wxyz(quat)
        rotation = np.asarray(
            self.data.geom_xmat[int(self.payload_geom_id)], dtype=np.float64
        ).reshape(3, 3)
        payload_size = np.asarray(
            self.model.geom_size[int(self.payload_geom_id)], dtype=np.float64
        )
        designated_top_surface_z_range_m = 2.0 * float(
            abs(rotation[2, 0]) * payload_size[0]
            + abs(rotation[2, 1]) * payload_size[1]
        )
        designated_top_face_tilt_deg = float(
            np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))
        )
        (
            support_face_tilt_deg,
            support_face_z_range_m,
            support_face_axis,
        ) = payload_support_face_attitude(rotation, payload_size)
        velocity = self.data.qvel[self.payload_dof_adr : self.payload_dof_adr + 6]
        support_target, support_wait_reasons = self._live_payload_support_target(
            require_ready=False
        )
        support_report: dict[str, object] = {
            "ready": support_target is not None and not support_wait_reasons,
            "tracking_available": support_target is not None,
            "wait_reasons": list(support_wait_reasons),
        }
        signed_longitudinal_offset_m: float | None = None
        signed_lateral_offset_m: float | None = None
        raw_center_signed_longitudinal_offset_m: float | None = None
        settled_rearward_offset_m: float | None = None
        if support_target is not None:
            robot_yaws = [
                _yaw_from_wxyz(
                    self.data.qpos[
                        slot.root_qpos_adr + 3 : slot.root_qpos_adr + 7
                    ]
                )
                for slot in self.slots
            ]
            mean_heading = _circular_mean(robot_yaws)
            offset_xy = np.asarray(qpos[:2] - support_target[:2], dtype=np.float64)
            body_forward = np.asarray(
                [np.cos(mean_heading), np.sin(mean_heading)],
                dtype=np.float64,
            )
            body_left = np.asarray(
                [-np.sin(mean_heading), np.cos(mean_heading)],
                dtype=np.float64,
            )
            signed_longitudinal_offset_m = float(np.dot(offset_xy, body_forward))
            signed_lateral_offset_m = float(np.dot(offset_xy, body_left))
            raw_center_signed_longitudinal_offset_m = (
                signed_longitudinal_offset_m - float(self.payload_rearward_offset_m)
            )
            settled_rearward_offset_m = -raw_center_signed_longitudinal_offset_m
            support_report.update(
                {
                    "x": float(support_target[0]),
                    "y": float(support_target[1]),
                    "z": float(support_target[2]),
                    "yaw_rad": _yaw_from_wxyz(support_target[3:7]),
                }
            )
        payload_body_id = int(self.model.geom_bodyid[int(self.payload_geom_id)])
        return {
            "present": True,
            "active": bool(self.payload_active),
            "mass_kg": float(self.model.body_mass[payload_body_id]),
            "x": float(qpos[0]),
            "y": float(qpos[1]),
            "z": float(qpos[2]),
            "quaternion_wxyz": [float(value) for value in quat],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "roll_deg": float(np.degrees(roll)),
            "pitch_deg": float(np.degrees(pitch)),
            "yaw_deg": float(np.degrees(yaw)),
            "half_extents_xyz_m": [float(value) for value in payload_size],
            "tilt_deg": support_face_tilt_deg,
            "top_surface_corner_z_range_m": support_face_z_range_m,
            "attitude_reference": "nearest_geometrically_equivalent_support_face",
            "support_face_axis": support_face_axis,
            "designated_top_face_tilt_deg": designated_top_face_tilt_deg,
            "designated_top_surface_corner_z_range_m": (
                designated_top_surface_z_range_m
            ),
            "linear_speed_mps": float(np.linalg.norm(velocity[:3])),
            "angular_speed_radps": float(np.linalg.norm(velocity[3:6])),
            "last_event_id": self._last_payload_event_id,
            "released": bool(self._payload_release_active),
            "rearward_offset_m": self.payload_rearward_offset_m,
            "support_center_signed_longitudinal_offset_m": signed_longitudinal_offset_m,
            "support_center_signed_lateral_offset_m": signed_lateral_offset_m,
            "raw_support_center_signed_longitudinal_offset_m": (
                raw_center_signed_longitudinal_offset_m
            ),
            "settled_rearward_offset_from_raw_center_m": settled_rearward_offset_m,
            "support_offset_convention": (
                "positive longitudinal is body-forward/outward; negative is rearward/toward torso; "
                "positive lateral is body-left"
            ),
            "live_support_target": support_report,
            "support_polygon": self._payload_support_polygon_report(qpos[:2]),
            "support_geometries": self._payload_support_geometry_report(),
            "contacts": self._payload_contact_report(),
        }

    def _payload_support_geometry_report(self) -> dict[str, list[dict[str, object]]]:
        report: dict[str, list[dict[str, object]]] = {}
        for robot_id, geom_ids in self.payload_support_surface_ids.items():
            items: list[dict[str, object]] = []
            for geom_id in geom_ids:
                rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                normal = rotation[:, 2]
                world_quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(world_quat, rotation.reshape(-1))
                half_extent_z = float(
                    np.dot(np.abs(rotation[2]), self.model.geom_size[geom_id])
                )
                items.append(
                    {
                        "name": self.model.geom(geom_id).name,
                        "position_xyz": [
                            float(value) for value in self.data.geom_xpos[geom_id]
                        ],
                        "world_quaternion_wxyz": [float(value) for value in world_quat],
                        "normal_xyz": [float(value) for value in normal],
                        "normal_tilt_deg": float(
                            np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0)))
                        ),
                        "top_z_m": float(self.data.geom_xpos[geom_id][2]) + half_extent_z,
                    }
                )
            report[robot_id] = items
        return report

    def _payload_support_polygon_report(
        self,
        payload_xy: np.ndarray,
    ) -> dict[str, object]:
        corners: list[np.ndarray] = []
        for geom_ids in self.payload_support_surface_ids.values():
            for geom_id in geom_ids:
                rotation = np.asarray(
                    self.data.geom_xmat[geom_id], dtype=np.float64
                ).reshape(3, 3)
                center = np.asarray(self.data.geom_xpos[geom_id][:2], dtype=np.float64)
                half_x, half_y = (
                    float(self.model.geom_size[geom_id][0]),
                    float(self.model.geom_size[geom_id][1]),
                )
                axis_x = rotation[:2, 0]
                axis_y = rotation[:2, 1]
                for sign_x in (-1.0, 1.0):
                    for sign_y in (-1.0, 1.0):
                        corners.append(
                            center + sign_x * half_x * axis_x + sign_y * half_y * axis_y
                        )
        hull = _convex_hull_xy(corners)
        margin_m = _signed_convex_polygon_margin_xy(
            np.asarray(payload_xy, dtype=np.float64),
            hull,
        )
        return {
            "available": margin_m is not None,
            "inside": margin_m is not None and margin_m >= 0.0,
            "com_margin_m": margin_m,
            "vertex_count": len(hull),
            "surface_count": sum(
                len(geom_ids) for geom_ids in self.payload_support_surface_ids.values()
            ),
            "model": "convex_hull_of_palm_and_forearm_support_surfaces",
        }

    def _payload_contact_report(self) -> dict[str, object]:
        if self.payload_geom_id is None:
            return {
                "robots": [],
                "support_robots": [],
                "support_geometries": [],
                "non_support_robot_contacts": [],
                "support_model": "hand_wrist_forearm_contacts",
                "walls": [],
                "floor": False,
                "target": False,
                "contact_count": 0,
                "min_dist": 0.0,
                "samples": [],
                "support_forces": {
                    "available": False,
                    "normal_force_n_by_robot": {},
                    "normal_force_n_by_geometry": {},
                    "total_normal_force_n": 0.0,
                    "min_robot_normal_force_n": 0.0,
                    "balance_ratio": 0.0,
                },
            }
        payload_geom_id = int(self.payload_geom_id)
        robots: set[str] = set()
        support_robots: set[str] = set()
        support_geometries: set[str] = set()
        non_support_robot_contacts: set[str] = set()
        walls: set[str] = set()
        floor = False
        target = False
        min_dist = float("inf")
        contact_count = 0
        samples: list[dict[str, object]] = []
        support_force_available = True
        support_force_error = ""
        support_force_by_robot = {"front": 0.0, "back": 0.0}
        support_force_by_geometry: dict[str, float] = {}
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if payload_geom_id not in (geom1, geom2):
                continue
            other_geom = geom2 if geom1 == payload_geom_id else geom1
            other_name = self.model.geom(other_geom).name
            other_body = int(self.model.geom_bodyid[other_geom])
            other_body_name = self.model.body(other_body).name
            contact_count += 1
            min_dist = min(min_dist, float(contact.dist))
            robot_id = self._robot_id_from_contact_name(other_name, other_body_name)
            support_normal_force_n: float | None = None
            if robot_id:
                robots.add(robot_id)
                if is_payload_upper_limb_support_contact(other_name, other_body_name):
                    support_label = other_name or other_body_name
                    support_robots.add(robot_id)
                    support_geometries.add(support_label)
                    contact_force = np.zeros(6, dtype=np.float64)
                    try:
                        mujoco.mj_contactForce(self.model, self.data, index, contact_force)
                    except Exception as exc:  # pragma: no cover - binding/runtime compatibility
                        support_force_available = False
                        support_force_error = f"{type(exc).__name__}: {exc}"
                    else:
                        measured_force_n = float(contact_force[0])
                        if (
                            not np.isfinite(measured_force_n)
                            or abs(measured_force_n) > MAX_RUNTIME_CONTACT_FORCE_N
                        ):
                            support_force_available = False
                            support_force_error = (
                                "invalid native support force at contact "
                                f"{index}: {measured_force_n!r}N"
                            )
                        else:
                            support_normal_force_n = max(0.0, measured_force_n)
                            support_force_by_robot[robot_id] += support_normal_force_n
                            support_force_by_geometry[support_label] = (
                                support_force_by_geometry.get(support_label, 0.0)
                                + support_normal_force_n
                            )
                else:
                    non_support_robot_contacts.add(other_name or other_body_name)
            if other_name.startswith("wall_") or other_body_name.startswith("wall_"):
                walls.add(other_name or other_body_name)
            if other_name in {"manual_floor", "floor"}:
                floor = True
            if other_name == "target_marker":
                target = True
            if len(samples) < 12:
                samples.append(
                    {
                        "geom": other_name,
                        "body": other_body_name,
                        "dist": float(contact.dist),
                        **(
                            {"support_normal_force_n": support_normal_force_n}
                            if support_normal_force_n is not None
                            else {}
                        ),
                    }
                )
        force_values = tuple(support_force_by_robot.values())
        total_support_force_n = sum(force_values)
        max_support_force_n = max(force_values, default=0.0)
        min_support_force_n = min(force_values, default=0.0)
        support_force_balance_ratio = (
            min_support_force_n / max_support_force_n
            if max_support_force_n > 1.0e-9
            else 0.0
        )
        return {
            "robots": sorted(robots),
            "support_robots": sorted(support_robots),
            "support_geometries": sorted(support_geometries),
            "non_support_robot_contacts": sorted(non_support_robot_contacts),
            "support_model": "hand_wrist_forearm_contacts",
            "walls": sorted(walls),
            "floor": bool(floor),
            "target": bool(target),
            "contact_count": contact_count,
            "min_dist": 0.0 if min_dist == float("inf") else min_dist,
            "samples": samples,
            "support_forces": {
                "available": support_force_available,
                "error": support_force_error,
                "normal_force_n_by_robot": support_force_by_robot,
                "normal_force_n_by_geometry": support_force_by_geometry,
                "total_normal_force_n": total_support_force_n,
                "min_robot_normal_force_n": min_support_force_n,
                "balance_ratio": support_force_balance_ratio,
            },
        }

    @staticmethod
    def _robot_id_from_contact_name(geom_name: str, body_name: str) -> str:
        for prefix, robot_id in (("front_", "front"), ("back_", "back")):
            if geom_name.startswith(prefix) or body_name.startswith(prefix):
                return robot_id
        return ""

    def _payload_target_qpos_from_site_or_joint(self) -> np.ndarray | None:
        if self.payload_support_site_id is not None:
            site_id = int(self.payload_support_site_id)
            return np.asarray(
                [
                    *self.model.site_pos[site_id],
                    *self.model.site_quat[site_id],
                ],
                dtype=np.float64,
            )
        if self.payload_qpos_adr is None:
            return None
        return self.data.qpos[self.payload_qpos_adr : self.payload_qpos_adr + 7].copy()

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

    def _optional_joint_id(self, name: str) -> int | None:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return None if joint_id < 0 else int(joint_id)

    def _optional_geom_id(self, name: str) -> int | None:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return None if geom_id < 0 else int(geom_id)

    def _optional_site_id(self, name: str) -> int | None:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return None if site_id < 0 else int(site_id)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Missing body: {name}")
        return int(body_id)


def _convex_hull_xy(points: list[np.ndarray]) -> list[np.ndarray]:
    unique = sorted(
        {(float(point[0]), float(point[1])) for point in points}
    )
    if len(unique) <= 1:
        return [np.asarray(point, dtype=np.float64) for point in unique]

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return [
        np.asarray(point, dtype=np.float64)
        for point in lower[:-1] + upper[:-1]
    ]


def _signed_convex_polygon_margin_xy(
    point: np.ndarray,
    polygon: list[np.ndarray],
) -> float | None:
    if len(polygon) < 3:
        return None
    margins: list[float] = []
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        edge = end - start
        edge_length = float(np.linalg.norm(edge))
        if edge_length <= 1.0e-9:
            continue
        margins.append(
            float(
                (edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0]))
                / edge_length
            )
        )
    return min(margins) if margins else None


def _yaw_from_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rpy_from_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quat)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def _quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def _circular_mean(angles: list[float]) -> float:
    if not angles:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))


def _payload_support_formation_wait_reasons(
    robot_centers: list[np.ndarray],
    robot_yaws: list[float],
) -> tuple[str, ...]:
    """Reject payload placement when the two robots no longer form a carry rack."""

    if len(robot_centers) != 2 or len(robot_yaws) != 2:
        return ("dual-robot carry formation is unavailable",)
    support_axis = np.asarray(robot_centers[1][:2] - robot_centers[0][:2], dtype=np.float64)
    separation = float(np.linalg.norm(support_axis))
    reasons: list[str] = []
    if separation < 0.70 or separation > 1.20:
        reasons.append(f"wrist support separation outside [0.70, 1.20]m ({separation:.3f}m)")
    if separation > 1.0e-9:
        support_yaw = float(np.arctan2(support_axis[1], support_axis[0]))
        mean_robot_yaw = _circular_mean(robot_yaws)
        axis_delta = abs(_wrapped_angle(support_yaw - mean_robot_yaw))
        perpendicular_error = abs(axis_delta - 0.5 * np.pi)
        if perpendicular_error > 0.25:
            reasons.append(
                "wrist support line is not perpendicular to robot heading "
                f"(error={perpendicular_error:.3f}rad)"
            )
    yaw_disagreement = abs(_wrapped_angle(robot_yaws[1] - robot_yaws[0]))
    if yaw_disagreement > 0.20:
        reasons.append(f"robot heading mismatch too large ({yaw_disagreement:.3f}rad)")
    return tuple(reasons)


def _wrapped_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(float(angle)), np.cos(float(angle))))


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


def _safe_checkpoint_id(value: str) -> str:
    cleaned = "".join(character for character in value.strip() if character.isalnum() or character in "-_.")
    if cleaned in {"", ".", ".."}:
        return ""
    return cleaned[:96]


def _planner_phase_is_idle(phase: str) -> bool:
    phase = str(phase).lower()
    return not phase or any(
        marker in phase
        for marker in ("idle", "stop_all", "checkpoint", "pre_stand", "safety_abort")
    )
