"""Physics feedback for bounded relative-formation alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import cos, sin
from pathlib import Path
import time


@dataclass
class RelativeAlignmentFeedback:
    """Stop one moving robot after measured body-frame catch-up."""

    runtime_state_file: Path
    mover_robot: str
    reference_robot: str
    axis: str
    signed_target_delta_m: float
    stop_margin_m: float = 0.02
    max_state_age_s: float = 1.0
    initial_poses: dict[str, tuple[float, float, float]] = field(init=False)
    motion_unit_xy: tuple[float, float] = field(init=False)
    target_distance_m: float = field(init=False)
    latest_progress_m: float = field(default=0.0, init=False)
    max_progress_m: float = field(default=0.0, init=False)
    sample_count: int = field(default=0, init=False)
    early_completion_count: int = field(default=0, init=False)
    last_error: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.runtime_state_file = Path(self.runtime_state_file)
        if {self.mover_robot, self.reference_robot} != {"front", "back"}:
            raise ValueError("relative alignment requires distinct front/back roles")
        if self.axis not in {"longitudinal", "lateral"}:
            raise ValueError(
                "relative alignment feedback supports longitudinal or lateral axes"
            )
        signed_target = float(self.signed_target_delta_m)
        if abs(signed_target) <= 1.0e-9:
            raise ValueError("relative alignment mover target must be nonzero")
        poses = self._read_poses()
        if poses is None:
            raise RuntimeError(
                "cannot initialize relative alignment feedback: "
                f"{self.last_error or self.runtime_state_file}"
            )
        self.initial_poses = poses
        mover_yaw = poses[self.mover_robot][2]
        if self.axis == "longitudinal":
            base_unit = (cos(mover_yaw), sin(mover_yaw))
        else:
            base_unit = (-sin(mover_yaw), cos(mover_yaw))
        sign = 1.0 if signed_target > 0.0 else -1.0
        self.motion_unit_xy = (sign * base_unit[0], sign * base_unit[1])
        self.target_distance_m = abs(signed_target)

    def progress_m(self) -> float | None:
        poses = self._read_poses()
        if poses is None:
            return None
        mover_now = poses[self.mover_robot]
        reference_now = poses[self.reference_robot]
        mover_initial = self.initial_poses[self.mover_robot]
        reference_initial = self.initial_poses[self.reference_robot]
        relative_dx = (
            mover_now[0]
            - mover_initial[0]
            - (reference_now[0] - reference_initial[0])
        )
        relative_dy = (
            mover_now[1]
            - mover_initial[1]
            - (reference_now[1] - reference_initial[1])
        )
        progress = (
            relative_dx * self.motion_unit_xy[0]
            + relative_dy * self.motion_unit_xy[1]
        )
        self.latest_progress_m = progress
        self.max_progress_m = max(self.max_progress_m, progress)
        self.sample_count += 1
        return progress

    def reached(self) -> bool:
        progress = self.progress_m()
        if progress is None:
            return False
        threshold = max(
            0.0,
            self.target_distance_m - max(0.0, float(self.stop_margin_m)),
        )
        reached = progress >= threshold
        if reached:
            self.early_completion_count += 1
        return reached

    def report(self) -> dict[str, object]:
        return {
            "enabled": True,
            "source": "mujoco_relative_robot_root_displacement",
            "mover_robot": self.mover_robot,
            "reference_robot": self.reference_robot,
            "axis": self.axis,
            "signed_target_delta_m": float(self.signed_target_delta_m),
            "target_distance_m": self.target_distance_m,
            "stop_margin_m": self.stop_margin_m,
            "motion_unit_xy": list(self.motion_unit_xy),
            "latest_progress_m": self.latest_progress_m,
            "max_progress_m": self.max_progress_m,
            "sample_count": self.sample_count,
            "early_completion_count": self.early_completion_count,
            "last_error": self.last_error,
        }

    def _read_poses(self) -> dict[str, tuple[float, float, float]] | None:
        try:
            state = json.loads(self.runtime_state_file.read_text(encoding="utf-8"))
            age_s = time.time() - float(state["time_unix"])
            if age_s > max(0.0, float(self.max_state_age_s)):
                raise RuntimeError(f"runtime state is stale by {age_s:.3f}s")
            robots = state["robots"]
            poses: dict[str, tuple[float, float, float]] = {}
            for robot_id in ("front", "back"):
                robot = robots[robot_id]
                root_pose = robot.get("root_pose", {})
                pose = root_pose if isinstance(root_pose, dict) and root_pose else robot
                poses[robot_id] = (
                    float(pose["x"]),
                    float(pose["y"]),
                    float(pose["yaw_rad"]),
                )
        except (
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self.last_error = str(exc)
            return None
        self.last_error = ""
        return poses


__all__ = ["RelativeAlignmentFeedback"]
