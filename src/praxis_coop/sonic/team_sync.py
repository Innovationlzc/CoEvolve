"""Low-rate formation feedback for two independent SONIC deploy backends.

The learned SONIC policies remain responsible for balance and joint control.
This module only adjusts their shared planner velocity envelope from live
MuJoCo root poses so one robot cannot run away from the other while carrying a
payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from math import cos, hypot, sin
from pathlib import Path
import time

from praxis_coop.sonic.commands import SonicCommand, SonicLocomotionMode


REQUIRED_TEAM = frozenset({"front", "back"})
SETTLE_STATIONARY_SAMPLES_REQUIRED = 3


@dataclass(frozen=True)
class TeamMotionSnapshot:
    """Atomic root/payload state used by the low-rate team controller."""

    available: bool = False
    fresh: bool = False
    age_s: float = float("inf")
    sim_time_s: float = 0.0
    robot_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    robot_yaw_rad: dict[str, float] = field(default_factory=dict)
    payload_xy: tuple[float, float] | None = None
    support_target_xy: tuple[float, float] | None = None
    heading_yaw_rad: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class FormationSyncConfig:
    """Conservative feedback limits for cooperative payload transport."""

    max_state_age_s: float = 1.0
    active_catchup: bool = False
    longitudinal_kp_s: float = 2.0
    cross_track_kp_s: float = 0.6
    payload_kp_s: float = 1.2
    error_filter_alpha: float = 0.35
    correction_deadband_m: float = 0.002
    pause_leader_error_m: float = 0.008
    resume_error_m: float = 0.003
    lagging_speed_boost_mps: float = 0.03
    max_correction_mps: float = 0.06
    # Leave headroom above the verified 0.50 command for bounded catch-up.
    max_command_speed_mps: float = 0.56
    min_active_speed_ratio: float = 0.0
    hard_desync_m: float | None = None


class FormationSyncController:
    """Keep front/back root progress and payload support offset synchronized."""

    def __init__(self, runtime_state_file: Path, config: FormationSyncConfig | None = None) -> None:
        self.runtime_state_file = Path(runtime_state_file)
        self.config = config or FormationSyncConfig()
        self._phase_id = ""
        self._phase_start_xy: dict[str, tuple[float, float]] = {}
        self._baseline_relative_xy: tuple[float, float] | None = None
        self._filtered_relative_error_xy = (0.0, 0.0)
        self._filtered_payload_offset_xy = (0.0, 0.0)
        self._last_snapshot_sim_time_s: float | None = None
        self._last_motion_robot_xy: dict[str, tuple[float, float]] = {}
        self._last_translation_commands: dict[str, SonicCommand] = {}
        self._last_translation_direction_world = (0.0, 0.0)
        self._settle_reference_xy: dict[str, tuple[float, float]] = {}
        self._settle_last_robot_xy: dict[str, tuple[float, float]] = {}
        self._settle_stationary_samples = 0
        self._settle_phase_id = ""
        self._catchup_leader = ""
        self._settle_catchup_leader = ""
        self._settle_catchup_completed = False
        self._settle_best_progress_gap_m: float | None = None
        self._endpoint_idle_locked = False
        self._sample_count = 0
        self._settle_sample_count = 0
        self._adjusted_sample_count = 0
        self._catchup_sample_count = 0
        self._catchup_activation_count = 0
        self._stale_sample_count = 0
        self._max_formation_error_m = 0.0
        self._max_longitudinal_gap_m = 0.0
        self._max_payload_support_offset_m = 0.0
        self._max_speed_correction_mps = 0.0
        self._max_publish_skew_ms = 0.0
        self._max_settle_residual_motion_m = 0.0
        self._last_diagnostics: dict[str, object] = {}

    def adjust(
        self,
        commands_by_robot: dict[str, SonicCommand],
    ) -> tuple[dict[str, SonicCommand], str]:
        snapshot = read_team_motion_snapshot(
            self.runtime_state_file,
            max_age_s=self.config.max_state_age_s,
        )
        return self.adjust_snapshot(commands_by_robot, snapshot)

    def adjust_snapshot(
        self,
        commands_by_robot: dict[str, SonicCommand],
        snapshot: TeamMotionSnapshot,
    ) -> tuple[dict[str, SonicCommand], str]:
        """Return corrected commands and an optional hard-stop reason."""

        if set(commands_by_robot) != REQUIRED_TEAM:
            return dict(commands_by_robot), ""
        reference = commands_by_robot["front"]
        if not _is_translation_command(reference):
            self._catchup_leader = ""
            if self._phase_id:
                return self._observe_settle_snapshot(
                    commands_by_robot,
                    snapshot,
                )
            return dict(commands_by_robot), ""

        self._sample_count += 1
        if not snapshot.available or not snapshot.fresh or not REQUIRED_TEAM.issubset(snapshot.robot_xy):
            self._stale_sample_count += 1
            self._last_diagnostics = {
                "phase_id": reference.phase_id,
                "state_available": snapshot.available,
                "state_fresh": snapshot.fresh,
                "state_age_s": snapshot.age_s,
                "error": snapshot.error,
            }
            return dict(commands_by_robot), ""

        direction = _unit_xy(reference.movement[:2])
        if reference.reference_frame == "body":
            direction = _rotate_xy(direction, snapshot.heading_yaw_rad)
        if direction == (0.0, 0.0):
            return dict(commands_by_robot), ""

        phase_id = reference.phase_id or reference.phase
        if phase_id != self._phase_id or not self._phase_start_xy:
            self._start_phase(phase_id, snapshot)
            self._last_translation_commands = dict(commands_by_robot)
            self._last_translation_direction_world = direction
            return dict(commands_by_robot), ""

        self._settle_reference_xy = {}
        self._settle_last_robot_xy = {}
        self._settle_stationary_samples = 0
        self._settle_phase_id = ""
        self._settle_catchup_leader = ""
        self._settle_catchup_completed = False
        self._settle_best_progress_gap_m = None
        self._last_motion_robot_xy = {
            robot_id: snapshot.robot_xy[robot_id]
            for robot_id in ("front", "back")
        }
        self._last_translation_commands = dict(commands_by_robot)
        self._last_translation_direction_world = direction

        current_relative = _sub(snapshot.robot_xy["front"], snapshot.robot_xy["back"])
        raw_relative_error = _sub(current_relative, self._baseline_relative_xy or current_relative)
        # This must remain absolute. Resetting it against every phase start made
        # an already displaced box the next phase's zero and allowed slip to
        # accumulate across otherwise accepted transactions.
        raw_payload_offset = _payload_offset(snapshot) or (0.0, 0.0)

        is_new_snapshot = snapshot.sim_time_s != self._last_snapshot_sim_time_s
        if is_new_snapshot:
            alpha = _clamp(self.config.error_filter_alpha, 0.0, 1.0)
            self._filtered_relative_error_xy = _blend(
                self._filtered_relative_error_xy,
                raw_relative_error,
                alpha,
            )
            self._filtered_payload_offset_xy = _blend(
                self._filtered_payload_offset_xy,
                raw_payload_offset,
                alpha,
            )
            self._last_snapshot_sim_time_s = snapshot.sim_time_s

        formation_error = self._filtered_relative_error_xy
        payload_offset = self._filtered_payload_offset_xy
        cross_track_direction = (-direction[1], direction[0])
        cross_track_error = _dot(formation_error, cross_track_direction)
        formation_error_m = _norm(formation_error)
        payload_offset_m = _norm(payload_offset)
        front_progress = _dot(
            _sub(snapshot.robot_xy["front"], self._phase_start_xy["front"]),
            direction,
        )
        back_progress = _dot(
            _sub(snapshot.robot_xy["back"], self._phase_start_xy["back"]),
            direction,
        )
        phase_progress_gap = front_progress - back_progress
        current_relative = _sub(
            snapshot.robot_xy["front"],
            snapshot.robot_xy["back"],
        )
        # Preserve the relative formation owned by the accepted checkpoint.
        # A turn can intentionally leave the two robots staggered along either
        # body axis. Treating that absolute stagger as zero-error would speed up
        # the already leading robot after a frame change. Endpoint alignment is
        # an explicit coordination action; this feedback loop only prevents new
        # front/back divergence during the current transaction.
        absolute_longitudinal_stagger = _dot(current_relative, direction)
        progress_gap = phase_progress_gap

        self._max_formation_error_m = max(self._max_formation_error_m, formation_error_m)
        self._max_longitudinal_gap_m = max(self._max_longitudinal_gap_m, abs(progress_gap))
        self._max_payload_support_offset_m = max(
            self._max_payload_support_offset_m,
            payload_offset_m,
        )

        # Pelvis progress is phase-dependent during humanoid walking: two healthy
        # SONIC policies can differ by several centimetres for one sample while
        # their palms and payload remain coupled.  A raw progress-gap stop turns
        # that ordinary gait phase offset into an emergency brake, which is more
        # likely to unload a palm than letting the short command finish.  Use the
        # filtered relative formation and payload/support drift for the live hard
        # stop; retain raw progress_gap as feedback and endpoint evidence.
        hard_error_m = max(formation_error_m, payload_offset_m)
        if (
            self.config.hard_desync_m is not None
            and hard_error_m > self.config.hard_desync_m
        ):
            reason = (
                "formation_sync_hard_limit: "
                f"error={hard_error_m:.4f}m filtered_error={formation_error_m:.4f}m "
                f"payload_support_offset={payload_offset_m:.4f}m "
                f"progress_gap={progress_gap:.4f}m limit={self.config.hard_desync_m:.4f}m "
                f"front_progress={front_progress:.4f}m back_progress={back_progress:.4f}m"
            )
            self._last_diagnostics = {
                "phase_id": phase_id,
                "failure_reason": reason,
                "formation_error_xy_m": list(formation_error),
                "longitudinal_gap_m": progress_gap,
                "payload_support_offset_xy_m": list(payload_offset),
            }
            return dict(commands_by_robot), reason

        # The verified loaded transport envelope uses one fixed, identical
        # command for both frozen SONIC policies. Updating target_vel on every
        # telemetry sample forces each independent planner to replan and can
        # disturb gait phase even when the correction is common to both robots.
        # Keep the live measurements as diagnostics in production; differential
        # catch-up remains an explicit ablation.
        base_speeds = {
            robot_id: max(0.0, float(command.speed))
            for robot_id, command in commands_by_robot.items()
        }
        if not self.config.active_catchup:
            self._last_diagnostics = {
                "phase_id": phase_id,
                "state_age_s": snapshot.age_s,
                "front_progress_m": front_progress,
                "back_progress_m": back_progress,
                "longitudinal_gap_m": progress_gap,
                "phase_progress_gap_m": phase_progress_gap,
                "absolute_longitudinal_stagger_m": (
                    absolute_longitudinal_stagger
                ),
                "formation_error_xy_m": list(formation_error),
                "payload_support_offset_xy_m": list(payload_offset),
                "catchup_leader": "",
                "catchup_lagger": "",
                "calibrated_base_speed_mps": base_speeds,
                "differential_correction_mps": 0.0,
                "payload_speed_correction_mps": 0.0,
                "commanded_speed_mps": base_speeds,
            }
            return dict(commands_by_robot), ""

        # Explicit catch-up experiments retain each calibrated baseline instead
        # of collapsing both robots to the smaller command.
        speed_targets = dict(base_speeds)
        differential_correction = 0.0
        leader_slowdown = 0.0
        lagger_speed_boost = 0.0
        catchup_lagger = ""
        # Independent SONIC policies exhibit centimetre-scale pelvis sway during
        # healthy walking. Differential correction is therefore opt-in. Payload
        # recentering below is always common to both commands and cannot create a
        # new front/back speed mismatch.
        self._update_catchup_state(progress_gap)
        if self._catchup_leader:
            requested_correction = max(
                0.0,
                self.config.longitudinal_kp_s
                * (abs(progress_gap) - self.config.resume_error_m),
            )
            lagger_speed_boost = min(
                self.config.lagging_speed_boost_mps,
                self.config.max_correction_mps,
                requested_correction,
            )
            # The lagging SONIC policy has only a small verified speed headroom,
            # but slowing the leader is intrinsically bounded by its existing
            # command. Keeping both corrections under the same small cap made a
            # 2:1 backend response mismatch impossible to recover: the leader
            # continued almost at full speed while the lagger saturated. Use the
            # full base-speed envelope for leader slowdown so "pause leader"
            # becomes literal when the measured progress gap is large.
            leader_slowdown = min(
                base_speeds[self._catchup_leader],
                requested_correction,
            )
            differential_correction = max(
                leader_slowdown,
                lagger_speed_boost,
            )
            catchup_lagger = (
                "back" if self._catchup_leader == "front" else "front"
            )
            leader_minimum_speed = (
                base_speeds[self._catchup_leader]
                * _clamp(self.config.min_active_speed_ratio, 0.0, 1.0)
            )
            speed_targets[self._catchup_leader] = max(
                leader_minimum_speed,
                base_speeds[self._catchup_leader] - leader_slowdown,
            )
            speed_targets[catchup_lagger] = min(
                self.config.max_command_speed_mps,
                base_speeds[catchup_lagger] + lagger_speed_boost,
            )
            self._catchup_sample_count += 1

        # A loaded long box cannot tolerate one robot stopping or steering
        # sideways while the other keeps walking. Preserve the shared planner
        # direction and apply only bounded longitudinal speed corrections.
        payload_longitudinal_error = _dot(payload_offset, direction)
        payload_speed_correction = _clamp(
            -self.config.payload_kp_s
            * _apply_scalar_deadband(
                payload_longitudinal_error,
                self.config.correction_deadband_m,
            ),
            -self.config.max_correction_mps,
            self.config.max_correction_mps,
        )
        if self._catchup_leader:
            # A common positive payload correction can accelerate the leader
            # back above its nominal speed and erase nearly all differential
            # authority. Formation recovery takes precedence until the
            # root-progress gap closes.
            payload_speed_correction = 0.0
        # Keep cross-track error as endpoint evidence only. Steering front and
        # back in opposite lateral directions unloads the two ends of a free
        # payload, even when that steering would reduce pelvis-relative error.
        # Loaded translation may vary longitudinal speed, but both SONIC
        # planners must retain the same semantic movement direction.
        cross_track_correction = 0.0

        adjusted: dict[str, SonicCommand] = {}
        commanded_speeds: dict[str, float] = {}
        commanded_movements: dict[str, tuple[float, float, float]] = {}
        for robot_id, command in commands_by_robot.items():
            minimum_active_speed = (
                base_speeds[robot_id]
                * _clamp(self.config.min_active_speed_ratio, 0.0, 1.0)
            )
            maximum_active_speed = max(
                base_speeds[robot_id],
                self.config.max_command_speed_mps,
            )
            speed = _clamp(
                speed_targets[robot_id] + payload_speed_correction,
                minimum_active_speed,
                maximum_active_speed,
            )
            movement = reference.movement
            adjusted[robot_id] = replace(
                command,
                movement=movement,
                facing=reference.facing,
                speed=speed,
            )
            commanded_speeds[robot_id] = speed
            commanded_movements[robot_id] = movement
            self._max_speed_correction_mps = max(
                self._max_speed_correction_mps,
                abs(speed - max(0.0, float(command.speed))),
            )

        if any(adjusted[item] != commands_by_robot[item] for item in adjusted):
            self._adjusted_sample_count += 1
        self._last_diagnostics = {
            "phase_id": phase_id,
            "state_age_s": snapshot.age_s,
            "front_progress_m": front_progress,
            "back_progress_m": back_progress,
            "longitudinal_gap_m": progress_gap,
            "phase_progress_gap_m": phase_progress_gap,
            "absolute_longitudinal_stagger_m": absolute_longitudinal_stagger,
            "formation_error_xy_m": list(formation_error),
            "payload_support_offset_xy_m": list(payload_offset),
            "cross_track_error_m": cross_track_error,
            "cross_track_correction_mps": cross_track_correction,
            "catchup_leader": self._catchup_leader,
            "catchup_lagger": catchup_lagger,
            "calibrated_base_speed_mps": base_speeds,
            "differential_correction_mps": differential_correction,
            "leader_slowdown_mps": leader_slowdown,
            "lagger_speed_boost_mps": lagger_speed_boost,
            "payload_speed_correction_mps": payload_speed_correction,
            "payload_correction_suppressed_for_catchup": bool(
                self._catchup_leader
            ),
            "commanded_speed_mps": commanded_speeds,
            "commanded_movement": commanded_movements,
        }
        return adjusted, ""

    def observe_publish_skew(self, skew_s: float) -> None:
        self._max_publish_skew_ms = max(self._max_publish_skew_ms, 1000.0 * max(0.0, skew_s))

    def lock_translation_endpoint(self) -> None:
        """Keep all subsequent settle commands idle after endpoint control stops."""

        self._endpoint_idle_locked = True
        self._settle_catchup_leader = ""
        self._settle_catchup_completed = True

    def report(self) -> dict[str, object]:
        return {
            "enabled": True,
            "controller": (
                "smooth_longitudinal_catchup"
                if self.config.active_catchup
                else "fixed_equal_command_monitor"
            ),
            "sample_count": self._sample_count,
            "settle_sample_count": self._settle_sample_count,
            "adjusted_sample_count": self._adjusted_sample_count,
            "catchup_sample_count": self._catchup_sample_count,
            "catchup_activation_count": self._catchup_activation_count,
            "stale_sample_count": self._stale_sample_count,
            "max_formation_error_m": self._max_formation_error_m,
            "max_longitudinal_gap_m": self._max_longitudinal_gap_m,
            "max_payload_support_offset_m": self._max_payload_support_offset_m,
            "max_speed_correction_mps": self._max_speed_correction_mps,
            "max_publish_skew_ms": self._max_publish_skew_ms,
            "max_settle_residual_motion_m": self._max_settle_residual_motion_m,
            "endpoint_idle_locked": self._endpoint_idle_locked,
            "last": self._last_diagnostics,
            "config": {
                key: value
                for key, value in vars(self.config).items()
            },
        }

    def _start_phase(self, phase_id: str, snapshot: TeamMotionSnapshot) -> None:
        self._phase_id = phase_id
        self._phase_start_xy = {
            robot_id: snapshot.robot_xy[robot_id]
            for robot_id in ("front", "back")
        }
        self._baseline_relative_xy = _sub(
            self._phase_start_xy["front"],
            self._phase_start_xy["back"],
        )
        self._filtered_relative_error_xy = (0.0, 0.0)
        self._filtered_payload_offset_xy = (0.0, 0.0)
        self._last_snapshot_sim_time_s = snapshot.sim_time_s
        self._last_motion_robot_xy = dict(self._phase_start_xy)
        self._settle_reference_xy = {}
        self._settle_last_robot_xy = {}
        self._settle_stationary_samples = 0
        self._settle_phase_id = ""
        self._catchup_leader = ""
        self._settle_catchup_leader = ""
        self._settle_catchup_completed = False
        self._settle_best_progress_gap_m = None
        self._endpoint_idle_locked = False
        self._last_diagnostics = {
            "phase_id": phase_id,
            "phase_initialized": True,
            "start_xy": {key: list(value) for key, value in self._phase_start_xy.items()},
            "initial_payload_support_offset_xy_m": list(
                _payload_offset(snapshot) or (0.0, 0.0)
            ),
        }

    def _observe_settle_snapshot(
        self,
        commands_by_robot: dict[str, SonicCommand],
        snapshot: TeamMotionSnapshot,
    ) -> tuple[dict[str, SonicCommand], str]:
        """Measure residual root motion after a shared translation command stops."""

        self._sample_count += 1
        self._settle_sample_count += 1
        reference = commands_by_robot["front"]
        settle_phase_id = reference.phase_id or reference.phase
        if (
            not snapshot.available
            or not snapshot.fresh
            or not REQUIRED_TEAM.issubset(snapshot.robot_xy)
        ):
            self._stale_sample_count += 1
            self._last_diagnostics = {
                "phase_id": self._phase_id,
                "settle_phase_id": settle_phase_id,
                "settle_observation": True,
                "state_available": snapshot.available,
                "state_fresh": snapshot.fresh,
                "state_age_s": snapshot.age_s,
                "error": snapshot.error,
            }
            return dict(commands_by_robot), ""

        if not self._settle_reference_xy:
            self._settle_reference_xy = {
                robot_id: self._last_motion_robot_xy.get(
                    robot_id,
                    snapshot.robot_xy[robot_id],
                )
                for robot_id in ("front", "back")
            }
            self._settle_last_robot_xy = dict(self._settle_reference_xy)
            self._settle_phase_id = settle_phase_id

        current_relative = _sub(
            snapshot.robot_xy["front"],
            snapshot.robot_xy["back"],
        )
        raw_relative_error = _sub(
            current_relative,
            self._baseline_relative_xy or current_relative,
        )
        raw_payload_offset = _payload_offset(snapshot) or (0.0, 0.0)
        is_new_snapshot = snapshot.sim_time_s != self._last_snapshot_sim_time_s
        incremental_motion_m = {
            robot_id: _norm(
                _sub(
                    snapshot.robot_xy[robot_id],
                    self._settle_last_robot_xy.get(
                        robot_id,
                        snapshot.robot_xy[robot_id],
                    ),
                )
            )
            for robot_id in ("front", "back")
        }
        if is_new_snapshot:
            alpha = _clamp(self.config.error_filter_alpha, 0.0, 1.0)
            self._filtered_relative_error_xy = _blend(
                self._filtered_relative_error_xy,
                raw_relative_error,
                alpha,
            )
            self._filtered_payload_offset_xy = _blend(
                self._filtered_payload_offset_xy,
                raw_payload_offset,
                alpha,
            )
            self._last_snapshot_sim_time_s = snapshot.sim_time_s
            self._settle_last_robot_xy = {
                robot_id: snapshot.robot_xy[robot_id]
                for robot_id in ("front", "back")
            }
            if max(incremental_motion_m.values(), default=0.0) <= (
                self.config.pause_leader_error_m
            ):
                self._settle_stationary_samples += 1
            else:
                self._settle_stationary_samples = 0

        residual_vectors = {
            robot_id: _sub(
                snapshot.robot_xy[robot_id],
                self._settle_reference_xy[robot_id],
            )
            for robot_id in ("front", "back")
        }
        residual_distances = {
            robot_id: _norm(vector)
            for robot_id, vector in residual_vectors.items()
        }
        maximum_residual = max(residual_distances.values(), default=0.0)
        formation_error_m = _norm(self._filtered_relative_error_xy)
        payload_offset_m = _norm(self._filtered_payload_offset_xy)
        hard_error_m = max(formation_error_m, payload_offset_m)
        self._max_formation_error_m = max(
            self._max_formation_error_m,
            formation_error_m,
        )
        self._max_payload_support_offset_m = max(
            self._max_payload_support_offset_m,
            payload_offset_m,
        )
        self._max_settle_residual_motion_m = max(
            self._max_settle_residual_motion_m,
            maximum_residual,
        )
        progress_gap = _dot(
            raw_relative_error,
            self._last_translation_direction_world,
        )
        adjusted = dict(commands_by_robot)
        catchup_lagger = ""
        catchup_regression = False
        if (
            self.config.active_catchup
            and not self._endpoint_idle_locked
            and self._last_translation_direction_world != (0.0, 0.0)
            and REQUIRED_TEAM.issubset(self._last_translation_commands)
        ):
            if (
                self._settle_catchup_leader
                and is_new_snapshot
                and self._settle_best_progress_gap_m is not None
                and abs(progress_gap)
                > self._settle_best_progress_gap_m
                + self.config.pause_leader_error_m
            ):
                catchup_regression = True
            self._update_settle_catchup_state(
                progress_gap,
                allow_activation=(
                    self._settle_stationary_samples
                    >= SETTLE_STATIONARY_SAMPLES_REQUIRED
                ),
            )
            if self._settle_catchup_leader:
                catchup_lagger = (
                    "back"
                    if self._settle_catchup_leader == "front"
                    else "front"
                )
                template = self._last_translation_commands[catchup_lagger]
                catchup_speed = min(
                    self.config.max_command_speed_mps,
                    self.config.max_correction_mps,
                    self.config.lagging_speed_boost_mps,
                )
                adjusted[catchup_lagger] = replace(
                    template,
                    speed=catchup_speed,
                    duration_s=reference.duration_s,
                    phase=f"{settle_phase_id}.catchup_{catchup_lagger}",
                    phase_id=f"{settle_phase_id}.catchup_{catchup_lagger}",
                )
                self._catchup_sample_count += 1
                self._adjusted_sample_count += 1
                self._max_speed_correction_mps = max(
                    self._max_speed_correction_mps,
                    catchup_speed,
                )

        self._last_diagnostics = {
            "phase_id": self._phase_id,
            "settle_phase_id": self._settle_phase_id,
            "settle_observation": True,
            "state_age_s": snapshot.age_s,
            "move_end_xy": {
                key: list(value)
                for key, value in self._settle_reference_xy.items()
            },
            "idle_current_xy": {
                key: list(snapshot.robot_xy[key])
                for key in ("front", "back")
            },
            "residual_motion_xy_m": {
                key: list(value)
                for key, value in residual_vectors.items()
            },
            "residual_motion_m": residual_distances,
            "maximum_residual_motion_m": maximum_residual,
            "incremental_motion_m": incremental_motion_m,
            "settle_stationary_samples": self._settle_stationary_samples,
            "formation_error_xy_m": list(self._filtered_relative_error_xy),
            "payload_support_offset_xy_m": list(
                self._filtered_payload_offset_xy
            ),
            "settle_progress_gap_m": progress_gap,
            "settle_catchup_leader": self._settle_catchup_leader,
            "settle_catchup_lagger": catchup_lagger,
            "settle_catchup_completed": self._settle_catchup_completed,
            "commands_remain_idle": not bool(catchup_lagger),
            "endpoint_idle_locked": self._endpoint_idle_locked,
        }
        if catchup_regression:
            warning = (
                "formation_sync_settle_catchup_regression: "
                f"progress_gap={progress_gap:.4f}m "
                f"best_gap={self._settle_best_progress_gap_m:.4f}m"
            )
            # Independent SONIC gait phases can overshoot by a few centimetres
            # while braking. Keep this as controller-quality evidence; only the
            # hard formation envelope below may abort an otherwise safe step.
            self._last_diagnostics["settle_catchup_regression_warning"] = True
            self._last_diagnostics["quality_warning"] = warning
        if (
            self.config.hard_desync_m is not None
            and hard_error_m > self.config.hard_desync_m
        ):
            reason = (
                "formation_sync_settle_hard_limit: "
                f"error={hard_error_m:.4f}m "
                f"filtered_error={formation_error_m:.4f}m "
                f"payload_support_offset={payload_offset_m:.4f}m "
                f"limit={self.config.hard_desync_m:.4f}m "
                f"front_residual={residual_distances['front']:.4f}m "
                f"back_residual={residual_distances['back']:.4f}m"
            )
            self._last_diagnostics["failure_reason"] = reason
            return dict(commands_by_robot), reason
        return adjusted, ""

    def _update_catchup_state(self, progress_gap: float) -> None:
        if self._catchup_leader:
            if abs(progress_gap) <= self.config.resume_error_m:
                self._catchup_leader = ""
            elif (
                self._catchup_leader == "front"
                and progress_gap < 0.0
            ) or (
                self._catchup_leader == "back"
                and progress_gap > 0.0
            ):
                # Do not reverse differential commands in the same telemetry
                # sample that crosses zero. Independent SONIC planners respond
                # with delay, so immediate leader swapping amplifies overshoot.
                # Clear first; the opposite side must exceed the wider
                # activation threshold on a later sample.
                self._catchup_leader = ""
            return
        if abs(progress_gap) <= self.config.pause_leader_error_m:
            return
        self._catchup_leader = "front" if progress_gap > 0.0 else "back"
        self._catchup_activation_count += 1

    def _update_settle_catchup_state(
        self,
        progress_gap: float,
        *,
        allow_activation: bool,
    ) -> None:
        """Apply one low-speed correction only after both backends settle."""

        if self._settle_catchup_leader:
            self._settle_best_progress_gap_m = min(
                abs(progress_gap),
                self._settle_best_progress_gap_m
                if self._settle_best_progress_gap_m is not None
                else abs(progress_gap),
            )
            if abs(progress_gap) <= self.config.resume_error_m:
                self._settle_catchup_leader = ""
                self._settle_catchup_completed = True
            elif (
                self._settle_catchup_leader == "front"
                and progress_gap < 0.0
            ) or (
                self._settle_catchup_leader == "back"
                and progress_gap > 0.0
            ):
                self._settle_catchup_leader = ""
                self._settle_catchup_completed = True
            return
        if (
            self._settle_catchup_completed
            or not allow_activation
            or abs(progress_gap) <= self.config.pause_leader_error_m
        ):
            return
        self._settle_catchup_leader = (
            "front" if progress_gap > 0.0 else "back"
        )
        self._settle_best_progress_gap_m = abs(progress_gap)
        self._catchup_activation_count += 1


def read_team_motion_snapshot(path: Path, *, max_age_s: float = 1.0) -> TeamMotionSnapshot:
    """Read one atomic runtime snapshot for cooperative velocity feedback."""

    path = Path(path)
    try:
        stat = path.stat()
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return TeamMotionSnapshot(error=f"runtime state unavailable at {path}: {exc}")
    if not isinstance(payload, dict):
        return TeamMotionSnapshot(error=f"runtime state at {path} is not a JSON object")

    robots = payload.get("robots", {})
    robot_xy: dict[str, tuple[float, float]] = {}
    robot_yaw_rad: dict[str, float] = {}
    if isinstance(robots, dict):
        for robot_id in ("front", "back"):
            state = robots.get(robot_id)
            if not isinstance(state, dict):
                continue
            xy = _optional_xy(state)
            if xy is not None:
                robot_xy[robot_id] = xy
            if "yaw_rad" in state:
                robot_yaw_rad[robot_id] = _safe_float(state.get("yaw_rad"))

    payload_state = payload.get("payload", {})
    if not isinstance(payload_state, dict):
        payload_state = {}
    payload_xy = _optional_xy(payload_state)
    target = payload_state.get("live_support_target", {})
    support_target_xy = _optional_xy(target) if isinstance(target, dict) else None
    age_s = max(0.0, time.time() - stat.st_mtime)
    return TeamMotionSnapshot(
        available=True,
        fresh=age_s <= max(0.0, float(max_age_s)),
        age_s=age_s,
        sim_time_s=_safe_float(payload.get("sim_time_s", 0.0)),
        robot_xy=robot_xy,
        robot_yaw_rad=robot_yaw_rad,
        payload_xy=payload_xy,
        support_target_xy=support_target_xy,
        heading_yaw_rad=_safe_float(payload.get("mean_heading_yaw_rad", 0.0)),
    )


def _is_translation_command(command: SonicCommand) -> bool:
    return (
        command.mode not in {SonicLocomotionMode.IDLE, SonicLocomotionMode.IDLE_SQUAT}
        and command.speed > 0.0
        and hypot(float(command.movement[0]), float(command.movement[1])) > 1.0e-6
        and command.motion_axis in {"longitudinal", "lateral"}
    )


def _payload_offset(snapshot: TeamMotionSnapshot) -> tuple[float, float] | None:
    if snapshot.payload_xy is None or snapshot.support_target_xy is None:
        return None
    return _sub(snapshot.support_target_xy, snapshot.payload_xy)


def _optional_xy(payload: dict[str, object]) -> tuple[float, float] | None:
    try:
        return float(payload["x"]), float(payload["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sub(left: tuple[float, float], right: tuple[float, float]) -> tuple[float, float]:
    return left[0] - right[0], left[1] - right[1]


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _norm(vector: tuple[float, float]) -> float:
    return hypot(vector[0], vector[1])


def _unit_xy(vector: tuple[float, float] | tuple[float, float, float]) -> tuple[float, float]:
    magnitude = hypot(float(vector[0]), float(vector[1]))
    if magnitude <= 1.0e-9:
        return 0.0, 0.0
    return float(vector[0]) / magnitude, float(vector[1]) / magnitude


def _apply_scalar_deadband(value: float, deadband: float) -> float:
    threshold = max(0.0, float(deadband))
    if abs(value) <= threshold:
        return 0.0
    return value - threshold if value > 0.0 else value + threshold


def _rotate_xy(value: tuple[float, float], yaw_rad: float) -> tuple[float, float]:
    c = cos(yaw_rad)
    s = sin(yaw_rad)
    return c * value[0] - s * value[1], s * value[0] + c * value[1]


def _blend(
    previous: tuple[float, float],
    current: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    return (
        (1.0 - alpha) * previous[0] + alpha * current[0],
        (1.0 - alpha) * previous[1] + alpha * current[1],
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
