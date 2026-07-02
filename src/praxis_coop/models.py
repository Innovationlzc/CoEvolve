"""Core typed schemas for Praxis-Coop.

The schemas are simulator-agnostic on purpose. Isaac, SONIC, and toy proxy
runtimes should adapt to these contracts instead of leaking runtime details into
the planner or evolution loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import atan2, cos, hypot, sin
from typing import Any


class SkillStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureMode(str, Enum):
    NONE = "none"
    CONTACT_LOSS = "contact_loss"
    COLLISION = "collision"
    CLEARANCE_VIOLATION = "clearance_violation"
    YAW_MISALIGNMENT = "yaw_misalignment"
    TARGET_MISS = "target_miss"
    STAGNATION = "stagnation"
    TIMEOUT = "timeout"
    ADAPTER_ERROR = "adapter_error"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0.0

    def distance_to(self, other: "Pose2D") -> float:
        return hypot(self.x - other.x, self.y - other.y)

    def translated(self, dx: float, dy: float, dyaw: float = 0.0) -> "Pose2D":
        return Pose2D(self.x + dx, self.y + dy, self.yaw + dyaw)


def wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


@dataclass
class RobotState:
    robot_id: str
    pose: Pose2D
    role: str = ""
    in_contact: bool = False
    local_obstacles: list[str] = field(default_factory=list)


@dataclass
class ObjectState:
    object_id: str
    pose: Pose2D
    size: tuple[float, float, float]
    tilt_roll: float = 0.0
    tilt_pitch: float = 0.0
    speed: float = 0.0


@dataclass
class GateState:
    gate_id: str
    center: Pose2D
    width: float
    axis_yaw: float


@dataclass
class TargetState:
    target_id: str
    pose: Pose2D
    tolerance_xy: float = 0.15
    tolerance_yaw: float = 0.15


@dataclass
class SceneGraph:
    scene_id: str
    robots: dict[str, RobotState]
    movable_object: ObjectState
    gate: GateState | None
    target: TargetState
    obstacles: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskTaxonomy:
    physical_coupling: str
    role_symmetry: str
    object_type: str
    environment_constraint: str
    observability: str
    objective: str
    interaction_type: str = "pure_cooperation"


@dataclass
class RoleAssignment:
    roles: dict[str, str]
    formation: str
    offsets: dict[str, Pose2D]


@dataclass
class TransportPlan:
    object_path: list[Pose2D]
    robot_paths: dict[str, list[Pose2D]]
    role_offsets: dict[str, Pose2D]
    sync_predicates: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContactIntent:
    """Contact-level intent attached to a skill phase.

    This is deliberately above joint control. It names what contact should be
    created or preserved so adapters can translate it to SONIC/WBC commands and
    verifiers can explain failures at the phase boundary.
    """

    object_id: str
    mode: str
    links: tuple[str, ...] = ("left_hand", "right_hand")
    target_region: str = ""
    maintain: bool = False
    force_range_n: tuple[float, float] | None = None


@dataclass(frozen=True)
class ContactFlowSpec:
    """Compact CFgen-lite contract for one cooperative skill node."""

    phase_id: str
    phase_index: int = 0
    contact: ContactIntent | None = None
    object_goal: Pose2D | None = None
    sync_group: str = ""
    abort_on_contact_loss: bool = False
    allowed_contact_loss_s: float = 0.0
    verifier_tags: tuple[str, ...] = ()


@dataclass
class SkillNode:
    name: str
    agents: dict[str, str]
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    monitors: list[str] = field(default_factory=list)
    fallback: list[str] = field(default_factory=list)
    timeout_s: float = 20.0
    phase_id: str = ""
    sync_group: str = ""
    contact_flow: ContactFlowSpec | None = None


@dataclass
class SkillGraph:
    graph_id: str
    nodes: list[SkillNode]
    taxonomy: TaskTaxonomy
    roles: RoleAssignment
    specs: list[str] = field(default_factory=list)


@dataclass
class SkillResult:
    node_name: str
    status: SkillStatus
    metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""
    phase_id: str = ""
    sync_group: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierReport:
    success: bool
    failure_mode: FailureMode = FailureMode.NONE
    metrics: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutableSpec:
    spec_id: str
    task_context: dict[str, str]
    rule_text: str
    trigger: list[str]
    executable_predicate: list[str]
    fallback: list[str]
    confidence: float = 0.5
    evidence: dict[str, list[str]] = field(
        default_factory=lambda: {"failure_cases": [], "success_after_patch": []}
    )


@dataclass
class PatchProposal:
    patch_id: str
    patch_type: str
    target: str
    change: dict[str, Any]
    rationale: str
