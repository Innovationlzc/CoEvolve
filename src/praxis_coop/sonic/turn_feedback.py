"""Physics feedback for bounded rigid-payload turn transactions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import atan2, cos, radians, sin
from pathlib import Path
import time


@dataclass
class PayloadYawFeedback:
    """Stop a commanded turn from measured payload yaw instead of elapsed time."""

    runtime_state_file: Path
    expected_direction: int
    stop_margin_rad: float = radians(0.75)
    max_state_age_s: float = 1.0
    initial_yaw_rad: float = field(init=False)
    latest_yaw_rad: float = field(init=False)
    max_progress_rad: float = field(default=0.0, init=False)
    sample_count: int = field(default=0, init=False)
    early_completion_count: int = field(default=0, init=False)
    last_error: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.runtime_state_file = Path(self.runtime_state_file)
        self.expected_direction = 1 if int(self.expected_direction) >= 0 else -1
        yaw = self._read_yaw()
        if yaw is None:
            raise RuntimeError(
                "cannot initialize payload yaw feedback: "
                f"{self.last_error or self.runtime_state_file}"
            )
        self.initial_yaw_rad = yaw
        self.latest_yaw_rad = yaw

    def progress_rad(self) -> float | None:
        yaw = self._read_yaw()
        if yaw is None:
            return None
        self.latest_yaw_rad = yaw
        self.sample_count += 1
        progress = self.expected_direction * _wrapped_angle(
            yaw - self.initial_yaw_rad
        )
        self.max_progress_rad = max(self.max_progress_rad, progress)
        return progress

    def reached(self, signed_target_yaw_rad: float) -> bool:
        target = self.expected_direction * float(signed_target_yaw_rad)
        if target <= 0.0:
            return False
        progress = self.progress_rad()
        if progress is None:
            return False
        reached = progress >= max(0.0, target - max(0.0, self.stop_margin_rad))
        if reached:
            self.early_completion_count += 1
        return reached

    def report(self) -> dict[str, object]:
        return {
            "enabled": True,
            "source": "mujoco_payload_yaw",
            "initial_yaw_rad": self.initial_yaw_rad,
            "latest_yaw_rad": self.latest_yaw_rad,
            "expected_direction": self.expected_direction,
            "max_progress_rad": self.max_progress_rad,
            "stop_margin_rad": self.stop_margin_rad,
            "sample_count": self.sample_count,
            "early_completion_count": self.early_completion_count,
            "last_error": self.last_error,
        }

    def _read_yaw(self) -> float | None:
        try:
            state = json.loads(self.runtime_state_file.read_text(encoding="utf-8"))
            age_s = time.time() - float(state["time_unix"])
            if age_s > max(0.0, float(self.max_state_age_s)):
                raise RuntimeError(f"runtime state is stale by {age_s:.3f}s")
            yaw = float(state["payload"]["yaw_rad"])
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
        return yaw


def _wrapped_angle(angle: float) -> float:
    return atan2(sin(float(angle)), cos(float(angle)))


__all__ = ["PayloadYawFeedback"]
