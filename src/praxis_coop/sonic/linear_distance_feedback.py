"""Measured endpoint feedback for low-rate SONIC translation commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import cos, hypot, isfinite, sin
from pathlib import Path
import time
from typing import Any


LINEAR_FEEDBACK_SCHEMA_VERSION = 1
DEFAULT_BRAKING_MARGIN_M = 0.05
MIN_BRAKING_MARGIN_M = 0.02
MAX_BRAKING_MARGIN_M = 0.20
LINEAR_FEEDBACK_TIMEOUT_GAIN = 2.0
MIN_CALIBRATION_SAMPLES = 2
MAX_CALIBRATION_SAMPLES_PER_KEY = 20


def linear_calibration_key(
    *,
    primitive: str,
    semantic_direction: str,
    loaded_context: str,
) -> str:
    """Return a non-transferable response-calibration key."""

    primitive = str(primitive).strip()
    direction = str(semantic_direction).strip().lower()
    loaded = str(loaded_context).strip().lower()
    if not primitive or direction not in {"forward", "backward", "left", "right"}:
        raise ValueError("linear calibration requires a primitive and semantic direction")
    if loaded not in {"loaded", "unloaded"}:
        raise ValueError("linear calibration context must be loaded or unloaded")
    return f"{primitive}|{direction}|{loaded}"


@dataclass
class LinearDistanceFeedback:
    """Stop one translation from measured payload/team displacement.

    This controller never chooses a route direction. It receives the already
    selected primitive and semantic direction, projects live MuJoCo motion onto
    that fixed direction, and terminates the command near the requested endpoint.
    """

    runtime_state_file: Path
    target_distance_m: float
    target_direction_xy: tuple[float, float]
    reference_frame: str
    primitive: str
    semantic_direction: str
    loaded_context: str
    command_speed_mps: float
    calibration_file: Path | None = None
    max_state_age_s: float = 1.0
    initial_state: dict[str, Any] = field(init=False)
    direction_world_xy: tuple[float, float] = field(init=False)
    calibration_key: str = field(init=False)
    braking_margin_m: float = field(init=False)
    calibration_sample_count: int = field(init=False)
    calibration_confident: bool = field(init=False)
    latest_payload_progress_m: float = field(default=0.0, init=False)
    maximum_payload_progress_m: float = field(default=0.0, init=False)
    trigger_payload_progress_m: float | None = field(default=None, init=False)
    latest_state: dict[str, Any] | None = field(default=None, init=False)
    trigger_state: dict[str, Any] | None = field(default=None, init=False)
    sample_count: int = field(default=0, init=False)
    stale_sample_count: int = field(default=0, init=False)
    completion_count: int = field(default=0, init=False)
    previous_speed_sample_state: dict[str, Any] = field(init=False)
    peak_payload_target_speed_mps: float = field(default=0.0, init=False)
    peak_team_mean_target_speed_mps: float = field(default=0.0, init=False)
    peak_robot_target_speed_mps: dict[str, float] = field(
        default_factory=lambda: {"front": 0.0, "back": 0.0}, init=False
    )
    last_error: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.runtime_state_file = Path(self.runtime_state_file)
        self.target_distance_m = float(self.target_distance_m)
        self.command_speed_mps = float(self.command_speed_mps)
        if not isfinite(self.target_distance_m) or self.target_distance_m <= 0.0:
            raise ValueError("linear distance target must be positive and finite")
        if not isfinite(self.command_speed_mps) or self.command_speed_mps <= 0.0:
            raise ValueError("linear command speed must be positive and finite")
        unit = _unit_xy(self.target_direction_xy)
        if unit is None:
            raise ValueError("linear distance feedback requires a nonzero direction")
        if self.reference_frame not in {"body", "world"}:
            raise ValueError("linear distance feedback reference frame must be body or world")
        self.calibration_key = linear_calibration_key(
            primitive=self.primitive,
            semantic_direction=self.semantic_direction,
            loaded_context=self.loaded_context,
        )
        initial = self._read_state()
        if initial is None:
            raise RuntimeError(
                "cannot initialize linear distance feedback: "
                f"{self.last_error or self.runtime_state_file}"
            )
        self.initial_state = initial
        self.previous_speed_sample_state = initial
        if self.reference_frame == "body":
            heading = float(initial["mean_heading_yaw_rad"])
            unit = (
                cos(heading) * unit[0] - sin(heading) * unit[1],
                sin(heading) * unit[0] + cos(heading) * unit[1],
            )
        self.direction_world_xy = unit
        estimate = braking_margin_estimate(
            self.calibration_file,
            calibration_key=self.calibration_key,
        )
        self.braking_margin_m = min(
            max(float(estimate["braking_margin_m"]), MIN_BRAKING_MARGIN_M),
            min(MAX_BRAKING_MARGIN_M, 0.5 * self.target_distance_m),
        )
        self.calibration_sample_count = int(estimate["sample_count"])
        self.calibration_confident = bool(estimate["confident"])

    @property
    def stop_threshold_m(self) -> float:
        return max(0.0, self.target_distance_m - self.braking_margin_m)

    def progress_m(self) -> float | None:
        state = self._read_state()
        if state is None:
            self.stale_sample_count += 1
            return None
        progress = _project_payload_delta(
            self.initial_state,
            state,
            self.direction_world_xy,
        )
        self.latest_state = state
        self._record_speed_sample(state)
        self.latest_payload_progress_m = progress
        self.maximum_payload_progress_m = max(self.maximum_payload_progress_m, progress)
        self.sample_count += 1
        return progress

    def reached(self) -> bool:
        progress = self.progress_m()
        if progress is None or progress < self.stop_threshold_m:
            return False
        if self.trigger_payload_progress_m is None:
            self.trigger_payload_progress_m = progress
            self.trigger_state = self.latest_state
        self.completion_count += 1
        return True

    def finalize(self, *, physical_transaction_completed: bool) -> dict[str, Any]:
        """Measure the settled endpoint and update only this exact response key."""

        final_state = self._read_state()
        if final_state is not None:
            self._record_speed_sample(final_state)
        final_progress = (
            None
            if final_state is None
            else _project_payload_delta(
                self.initial_state,
                final_state,
                self.direction_world_xy,
            )
        )
        braking_drift = (
            None
            if final_progress is None or self.trigger_payload_progress_m is None
            else final_progress - self.trigger_payload_progress_m
        )
        endpoint_error = (
            None if final_progress is None else final_progress - self.target_distance_m
        )
        trigger_motion = (
            None
            if self.trigger_state is None
            else _measured_motion(
                self.initial_state,
                self.trigger_state,
                self.direction_world_xy,
            )
        )
        settled_motion = (
            None
            if final_state is None
            else _measured_motion(
                self.initial_state,
                final_state,
                self.direction_world_xy,
            )
        )
        sample_written = False
        if (
            physical_transaction_completed
            and self.calibration_file is not None
            and final_progress is not None
            and self.trigger_payload_progress_m is not None
            and braking_drift is not None
            and isfinite(braking_drift)
        ):
            append_linear_calibration_sample(
                self.calibration_file,
                calibration_key=self.calibration_key,
                sample={
                    "recorded_at_unix": time.time(),
                    "primitive": self.primitive,
                    "semantic_direction": self.semantic_direction,
                    "loaded_context": self.loaded_context,
                    "command_speed_mps": self.command_speed_mps,
                    "target_distance_m": self.target_distance_m,
                    "stop_threshold_m": self.stop_threshold_m,
                    "trigger_payload_progress_m": self.trigger_payload_progress_m,
                    "settled_payload_progress_m": final_progress,
                    "braking_drift_m": braking_drift,
                    "endpoint_error_m": endpoint_error,
                },
            )
            sample_written = True
        return {
            **self.report(),
            "physical_transaction_completed": bool(physical_transaction_completed),
            "settled_payload_progress_m": final_progress,
            "braking_drift_m": braking_drift,
            "endpoint_error_m": endpoint_error,
            "move_end_measured_motion": trigger_motion,
            "settled_measured_motion": settled_motion,
            "calibration_sample_written": sample_written,
        }

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "source": "mujoco_payload_projected_displacement",
            "primitive": self.primitive,
            "semantic_direction": self.semantic_direction,
            "loaded_context": self.loaded_context,
            "calibration_key": self.calibration_key,
            "calibration_file": (
                "" if self.calibration_file is None else str(self.calibration_file)
            ),
            "calibration_sample_count": self.calibration_sample_count,
            "calibration_confident": self.calibration_confident,
            "command_speed_mps": self.command_speed_mps,
            "target_distance_m": self.target_distance_m,
            "braking_margin_m": self.braking_margin_m,
            "stop_threshold_m": self.stop_threshold_m,
            "reference_frame": self.reference_frame,
            "target_direction_world_xy": list(self.direction_world_xy),
            "latest_payload_progress_m": self.latest_payload_progress_m,
            "maximum_payload_progress_m": self.maximum_payload_progress_m,
            "trigger_payload_progress_m": self.trigger_payload_progress_m,
            "peak_measured_target_direction_speed_mps": (
                self.peak_payload_target_speed_mps
            ),
            "peak_team_mean_target_direction_speed_mps": (
                self.peak_team_mean_target_speed_mps
            ),
            "peak_robot_target_direction_speed_mps": dict(
                self.peak_robot_target_speed_mps
            ),
            "sample_count": self.sample_count,
            "stale_sample_count": self.stale_sample_count,
            "completion_count": self.completion_count,
            "last_error": self.last_error,
        }

    def _record_speed_sample(self, state: dict[str, Any]) -> None:
        speeds = _instantaneous_target_speeds(
            self.previous_speed_sample_state,
            state,
            self.direction_world_xy,
        )
        self.previous_speed_sample_state = state
        if speeds is None:
            return
        self.peak_payload_target_speed_mps = max(
            self.peak_payload_target_speed_mps,
            float(speeds["payload"]),
        )
        self.peak_team_mean_target_speed_mps = max(
            self.peak_team_mean_target_speed_mps,
            float(speeds["team_mean"]),
        )
        robots = speeds["robots"]
        for robot_id in ("front", "back"):
            self.peak_robot_target_speed_mps[robot_id] = max(
                self.peak_robot_target_speed_mps[robot_id],
                float(robots[robot_id]),
            )

    def _read_state(self) -> dict[str, Any] | None:
        try:
            stat = self.runtime_state_file.stat()
            state = json.loads(self.runtime_state_file.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise RuntimeError("runtime state is not a JSON object")
            age_s = max(0.0, time.time() - stat.st_mtime)
            if age_s > max(0.0, float(self.max_state_age_s)):
                raise RuntimeError(f"runtime state is stale by {age_s:.3f}s")
            payload = state["payload"]
            mean_heading = float(state["mean_heading_yaw_rad"])
            result = {
                "sim_time_s": float(state["sim_time_s"]),
                "mean_heading_yaw_rad": mean_heading,
                "payload": {"x": float(payload["x"]), "y": float(payload["y"])},
                "robots": {
                    robot_id: {
                        "x": float(state["robots"][robot_id]["x"]),
                        "y": float(state["robots"][robot_id]["y"]),
                    }
                    for robot_id in ("front", "back")
                },
            }
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
        return result


def braking_margin_estimate(
    path: Path | None,
    *,
    calibration_key: str,
) -> dict[str, Any]:
    payload = _read_store(path)
    entries = payload.get("calibrations", {}).get(calibration_key, [])
    entries = entries if isinstance(entries, list) else []
    drifts = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            value = float(entry["braking_drift_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if isfinite(value):
            drifts.append(max(0.0, value))
    recent = drifts[-MAX_CALIBRATION_SAMPLES_PER_KEY:]
    confident = len(recent) >= MIN_CALIBRATION_SAMPLES
    margin = DEFAULT_BRAKING_MARGIN_M
    if confident:
        ordered = sorted(recent)
        # A bounded upper-median estimate avoids one noisy rollout taking over
        # while still stopping before the typical settled drift.
        margin = ordered[len(ordered) // 2]
    return {
        "braking_margin_m": min(
            max(margin, MIN_BRAKING_MARGIN_M), MAX_BRAKING_MARGIN_M
        ),
        "sample_count": len(recent),
        "confident": confident,
        "minimum_samples": MIN_CALIBRATION_SAMPLES,
    }


def append_linear_calibration_sample(
    path: Path,
    *,
    calibration_key: str,
    sample: dict[str, Any],
) -> None:
    path = Path(path)
    payload = _read_store(path)
    calibrations = payload.setdefault("calibrations", {})
    if not isinstance(calibrations, dict):
        calibrations = {}
        payload["calibrations"] = calibrations
    entries = calibrations.setdefault(calibration_key, [])
    if not isinstance(entries, list):
        entries = []
    entries.append(dict(sample))
    calibrations[calibration_key] = entries[-MAX_CALIBRATION_SAMPLES_PER_KEY:]
    payload.update(
        {
            "schema_version": LINEAR_FEEDBACK_SCHEMA_VERSION,
            "kind": "run_local_linear_response_calibration",
            "updated_at_unix": time.time(),
            "scope_contract": (
                "samples are isolated by primitive, semantic direction, and loaded context"
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_store(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {
            "schema_version": LINEAR_FEEDBACK_SCHEMA_VERSION,
            "kind": "run_local_linear_response_calibration",
            "calibrations": {},
        }
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": LINEAR_FEEDBACK_SCHEMA_VERSION,
            "kind": "run_local_linear_response_calibration",
            "calibrations": {},
        }
    return payload if isinstance(payload, dict) else {"calibrations": {}}


def _project_payload_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    direction: tuple[float, float],
) -> float:
    before_payload = before["payload"]
    after_payload = after["payload"]
    return (
        (float(after_payload["x"]) - float(before_payload["x"])) * direction[0]
        + (float(after_payload["y"]) - float(before_payload["y"])) * direction[1]
    )


def _measured_motion(
    before: dict[str, Any],
    after: dict[str, Any],
    direction: tuple[float, float],
) -> dict[str, Any]:
    elapsed_s = max(0.0, float(after["sim_time_s"]) - float(before["sim_time_s"]))

    def projected(point_before: dict[str, Any], point_after: dict[str, Any]) -> dict[str, float]:
        delta_x = float(point_after["x"]) - float(point_before["x"])
        delta_y = float(point_after["y"]) - float(point_before["y"])
        progress = delta_x * direction[0] + delta_y * direction[1]
        return {
            "delta_x_m": delta_x,
            "delta_y_m": delta_y,
            "net_displacement_m": hypot(delta_x, delta_y),
            "target_direction_progress_m": progress,
            "average_target_speed_mps": progress / elapsed_s if elapsed_s > 1.0e-9 else 0.0,
        }

    payload = projected(before["payload"], after["payload"])
    robots = {
        robot_id: projected(before["robots"][robot_id], after["robots"][robot_id])
        for robot_id in ("front", "back")
    }
    team_progress = sum(
        item["target_direction_progress_m"] for item in robots.values()
    ) / len(robots)
    return {
        "elapsed_sim_s": elapsed_s,
        "payload": payload,
        "robots": robots,
        "team_mean_target_direction_progress_m": team_progress,
        "robot_target_progress_gap_m": abs(
            robots["front"]["target_direction_progress_m"]
            - robots["back"]["target_direction_progress_m"]
        ),
    }


def _instantaneous_target_speeds(
    before: dict[str, Any],
    after: dict[str, Any],
    direction: tuple[float, float],
) -> dict[str, Any] | None:
    elapsed_s = float(after["sim_time_s"]) - float(before["sim_time_s"])
    if elapsed_s <= 1.0e-9:
        return None

    def speed(point_before: dict[str, Any], point_after: dict[str, Any]) -> float:
        delta_x = float(point_after["x"]) - float(point_before["x"])
        delta_y = float(point_after["y"]) - float(point_before["y"])
        return max(0.0, (delta_x * direction[0] + delta_y * direction[1]) / elapsed_s)

    payload_speed = speed(before["payload"], after["payload"])
    robot_speeds = {
        robot_id: speed(before["robots"][robot_id], after["robots"][robot_id])
        for robot_id in ("front", "back")
    }
    return {
        "payload": payload_speed,
        "robots": robot_speeds,
        "team_mean": sum(robot_speeds.values()) / len(robot_speeds),
    }


def _unit_xy(value: tuple[float, float]) -> tuple[float, float] | None:
    x, y = float(value[0]), float(value[1])
    norm = hypot(x, y)
    if not isfinite(norm) or norm <= 1.0e-9:
        return None
    return x / norm, y / norm


__all__ = [
    "DEFAULT_BRAKING_MARGIN_M",
    "LINEAR_FEEDBACK_SCHEMA_VERSION",
    "LINEAR_FEEDBACK_TIMEOUT_GAIN",
    "LinearDistanceFeedback",
    "append_linear_calibration_sample",
    "braking_margin_estimate",
    "linear_calibration_key",
]
