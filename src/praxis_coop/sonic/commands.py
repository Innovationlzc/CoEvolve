"""Typed command envelopes for SONIC / GEAR-SONIC planner control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum


class SonicLocomotionMode(IntEnum):
    """Subset of GEAR-SONIC LocalMotionPlanner LocomotionMode."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    OBJECT_CARRYING = 21


@dataclass(frozen=True)
class SonicCommand:
    """One planner command sent to a single SONIC deploy instance."""

    t: float
    robot_id: str
    skill_node: str
    mode: SonicLocomotionMode
    movement: tuple[float, float, float]
    facing: tuple[float, float, float]
    speed: float = -1.0
    height: float = -1.0
    duration_s: float = 0.2
    phase: str = ""
    phase_id: str = ""
    start_xy: tuple[float, float] | None = None
    target_xy: tuple[float, float] | None = None
    object_goal_xy: tuple[float, float] | None = None
    hand_pose_policy: str = "none"
    hand_reference: str = ""
    left_hand_joints: tuple[float, ...] | None = None
    right_hand_joints: tuple[float, ...] | None = None
    upper_body_position: tuple[float, ...] | None = None
    upper_body_velocity: tuple[float, ...] | None = None
    vr_position: tuple[float, ...] | None = None
    vr_orientation: tuple[float, ...] | None = None
    contact_intent: str = "none"
    contact_object_id: str = ""
    contact_links: tuple[str, ...] = field(default_factory=tuple)
    contact_target_region: str = ""
    sync_group: str = ""
    abort_on_contact_loss: bool = False
    allowed_contact_loss_s: float = 0.0
    motion_axis: str = ""
    reference_frame: str = "world"
    delta_yaw_rad: float | None = None
    target_yaw_rad: float | None = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["mode"] = int(self.mode)
        return payload


@dataclass
class SonicCommandStream:
    """A timestamped multi-robot SONIC command stream."""

    commands: list[SonicCommand] = field(default_factory=list)

    def extend(self, commands: list[SonicCommand]) -> None:
        self.commands.extend(commands)

    def append(self, command: SonicCommand) -> None:
        self.commands.append(command)

    def sorted(self) -> "SonicCommandStream":
        return SonicCommandStream(sorted(self.commands, key=lambda item: (item.t, item.robot_id)))

    def by_robot(self) -> dict[str, list[SonicCommand]]:
        grouped: dict[str, list[SonicCommand]] = {}
        for command in self.sorted().commands:
            grouped.setdefault(command.robot_id, []).append(command)
        return grouped

    @property
    def duration_s(self) -> float:
        if not self.commands:
            return 0.0
        return max(command.t + command.duration_s for command in self.commands)

    def to_json(self) -> list[dict]:
        return [command.to_json() for command in self.sorted().commands]
