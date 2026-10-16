"""Weak planar hold for stationary SONIC posture skills.

The frozen deploy planner stays responsible for balance. This helper only
converts measured root drift into a bounded planner target_vel in the existing
deploy frame. It never writes MuJoCo qpos or teleports the payload.

Official GEAR-SONIC static modes (IDLE / squat) overwrite planner speed to
``-1``, which means "use the mode default" rather than an explicit velocity.
The equivalent stationary command is therefore IDLE with ``movement=(0,0,0)``.
When root drift exceeds a deadband, this helper temporarily uses SLOW_WALK with
a small positive speed so the frozen controller can walk back onto the captured
standing XY.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import cos, hypot, sin
from pathlib import Path

from praxis_coop.sonic.commands import SonicCommand, SonicLocomotionMode
from praxis_coop.sonic.team_sync import TeamMotionSnapshot, read_team_motion_snapshot


PLACE_CROUCH_STATIONARY_LOCK_S = 0.5
PLACE_CROUCH_STATIONARY_SETTLE_S = 0.8
PLACE_CROUCH_STATIONARY_SPEED = -1.0
PLACE_CROUCH_HOLD_ENTER_DEADBAND_M = 0.012
PLACE_CROUCH_HOLD_EXIT_DEADBAND_M = 0.006
PLACE_CROUCH_HOLD_KP_S = 1.5
PLACE_CROUCH_HOLD_MAX_SPEED_MPS = 0.05


@dataclass
class StationaryPlanarHold:
    """Hold each robot's captured XY using official planner movement/speed."""

    runtime_state_file: Path | None = None
    enter_deadband_m: float = PLACE_CROUCH_HOLD_ENTER_DEADBAND_M
    exit_deadband_m: float = PLACE_CROUCH_HOLD_EXIT_DEADBAND_M
    kp_s: float = PLACE_CROUCH_HOLD_KP_S
    max_speed_mps: float = PLACE_CROUCH_HOLD_MAX_SPEED_MPS
    max_state_age_s: float = 1.0
    _reference_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    _correcting: dict[str, bool] = field(default_factory=dict)
    last_diagnostics: dict[str, object] = field(default_factory=dict)

    def capture(self, snapshot: TeamMotionSnapshot | None = None) -> bool:
        snapshot = snapshot or self._read_snapshot()
        if snapshot is None or not snapshot.available or not snapshot.fresh or not snapshot.robot_xy:
            self.last_diagnostics = {
                "captured": False,
                "error": (
                    snapshot.error
                    if snapshot is not None
                    else "runtime snapshot unavailable"
                ),
            }
            return False
        self._reference_xy = {
            robot_id: xy for robot_id, xy in snapshot.robot_xy.items()
        }
        self._correcting = {robot_id: False for robot_id in self._reference_xy}
        self.last_diagnostics = {
            "captured": True,
            "reference_xy": {
                robot_id: [float(xy[0]), float(xy[1])]
                for robot_id, xy in self._reference_xy.items()
            },
        }
        return True

    def adjust(
        self,
        commands_by_robot: dict[str, SonicCommand],
        snapshot: TeamMotionSnapshot | None = None,
    ) -> dict[str, SonicCommand]:
        if not self._reference_xy:
            return dict(commands_by_robot)
        snapshot = snapshot or self._read_snapshot()
        if snapshot is None or not snapshot.available or not snapshot.fresh:
            self.last_diagnostics = {
                "captured": True,
                "active": False,
                "error": (
                    snapshot.error
                    if snapshot is not None
                    else "stale runtime snapshot"
                ),
            }
            return {
                robot_id: _stationary_command(command)
                for robot_id, command in commands_by_robot.items()
            }

        adjusted: dict[str, SonicCommand] = {}
        robot_diag: dict[str, object] = {}
        for robot_id, command in commands_by_robot.items():
            current = snapshot.robot_xy.get(robot_id)
            reference = self._reference_xy.get(robot_id)
            yaw = float(snapshot.robot_yaw_rad.get(robot_id, 0.0))
            if current is None or reference is None:
                adjusted[robot_id] = _stationary_command(command)
                continue
            error_world = (reference[0] - current[0], reference[1] - current[1])
            error_m = hypot(error_world[0], error_world[1])
            correcting = bool(self._correcting.get(robot_id, False))
            if correcting:
                correcting = error_m > self.exit_deadband_m
            else:
                correcting = error_m > self.enter_deadband_m
            self._correcting[robot_id] = correcting
            if not correcting:
                adjusted[robot_id] = _stationary_command(command)
                robot_diag[robot_id] = {
                    "error_m": error_m,
                    "correction_mps": 0.0,
                    "mode": int(SonicLocomotionMode.IDLE),
                }
                continue
            body_forward, body_left = _world_error_to_deploy_frame(error_world, yaw)
            speed = min(self.max_speed_mps, self.kp_s * error_m)
            movement = _unit_xy((body_forward, body_left))
            adjusted[robot_id] = replace(
                command,
                mode=SonicLocomotionMode.SLOW_WALK,
                movement=(movement[0], movement[1], 0.0),
                speed=speed,
            )
            robot_diag[robot_id] = {
                "error_m": error_m,
                "body_forward_m": body_forward,
                "body_left_m": body_left,
                "correction_mps": speed,
                "mode": int(SonicLocomotionMode.SLOW_WALK),
            }
        self.last_diagnostics = {
            "captured": True,
            "active": True,
            "robots": robot_diag,
        }
        return adjusted

    def _read_snapshot(self) -> TeamMotionSnapshot | None:
        if self.runtime_state_file is None:
            return None
        return read_team_motion_snapshot(
            self.runtime_state_file,
            max_age_s=self.max_state_age_s,
        )


def _stationary_command(command: SonicCommand) -> SonicCommand:
    return replace(
        command,
        mode=SonicLocomotionMode.IDLE,
        movement=(0.0, 0.0, 0.0),
        speed=PLACE_CROUCH_STATIONARY_SPEED,
    )


def _world_error_to_deploy_frame(
    error_world: tuple[float, float],
    yaw_rad: float,
) -> tuple[float, float]:
    """Map world XY error into SONIC deploy-frame forward/left."""

    c = cos(yaw_rad)
    s = sin(yaw_rad)
    forward = c * error_world[0] + s * error_world[1]
    left = -s * error_world[0] + c * error_world[1]
    return forward, left


def _unit_xy(vector: tuple[float, float]) -> tuple[float, float]:
    magnitude = hypot(vector[0], vector[1])
    if magnitude <= 1.0e-9:
        return 0.0, 0.0
    return vector[0] / magnitude, vector[1] / magnitude
