"""Typed contracts for Code-as-Policy box-carry experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


FailureType = Literal[
    "none",
    "robot_fall",
    "payload_drop",
    "payload_not_supported",
    "payload_unstable",
    "payload_slip",
    "formation_desync",
    "payload_wall_contact",
    "payload_not_active",
    "no_payload_progress",
    "goal_regression",
    "route_regression",
    "straightness_regression",
    "alignment_regression",
    "payload_not_on_target",
    "robots_not_clear",
    "timeout",
    "script_error",
    "runtime_state_missing",
    "manual_review_required",
]

CoordinationMode = Literal["sync", "async"]
FailureLevel = Literal[
    "none",
    "strategy",
    "coordination",
    "script",
    "execution",
    "system",
]

DEFAULT_TARGET_XY_TOLERANCE_M = 0.25
DEFAULT_ROBOT_EXIT_MIN_DISTANCE_M = 1.15
DEFAULT_MAX_TERMINAL_PAYLOAD_TILT_DEG = 20.0
# Standing recovery is verifier-checked after execution. Requiring an extra
# retreat reserve before that probe can deadlock a crouched robot against a
# wall even after it has reached the task's final measurable clearance.
DEFAULT_ROBOT_PRE_STAND_CLEARANCE_MARGIN_M = 0.0


@dataclass(frozen=True)
class RouteProgressGate:
    """One scene-authored oriented gate in a route-topology specification.

    The algorithms consume only a normal vector and threshold. Scene axes,
    corridor names, and turn counts remain data owned by the task definition.
    """

    phase_name: str
    normal_x: float
    normal_y: float
    threshold_m: float


@dataclass(frozen=True)
class RouteSegment:
    """One connected route segment, independent of robot action semantics."""

    segment_id: str
    direction_x: float
    direction_y: float
    start_x_m: float
    start_y_m: float
    goal_x_m: float
    goal_y_m: float
    center_x_m: float
    center_y_m: float
    half_width_m: float
    detour_margin_m: float = 0.35


DEFAULT_MANUAL_LAYOUT_ROUTE_GATES = (
    RouteProgressGate("first_corner", 1.0, 0.0, 1.15),
    RouteProgressGate("cross_passage", 0.0, 1.0, 2.85),
    RouteProgressGate("second_corner", 0.0, 1.0, 7.35),
    RouteProgressGate("final_straight", -1.0, 0.0, -0.85),
)


DEFAULT_MANUAL_LAYOUT_ROUTE_SEGMENTS = (
    RouteSegment(
        "initial_straight", 1.0, 0.0, -1.55, 2.28, 1.35, 2.35, -0.10, 2.32, 1.45
    ),
    RouteSegment(
        "first_corner_entry", 0.0, 1.0, 1.35, 2.35, 1.35, 2.85, 1.35, 2.60, 1.30
    ),
    RouteSegment(
        "cross_passage", 0.0, 1.0, 1.35, 2.85, 1.35, 7.35, 1.35, 5.10, 1.30
    ),
    RouteSegment(
        "second_corner_entry", -1.0, 0.0, 1.35, 7.35, 0.85, 7.60, 1.10, 7.48, 1.45
    ),
    RouteSegment(
        "final_straight", -1.0, 0.0, 0.85, 7.60, -1.50, 7.60, -0.33, 7.60, 1.45
    ),
)


@dataclass(frozen=True)
class BoxCarryTask:
    """Human target and physical success boundaries for the long-box task."""

    task_id: str
    target: str
    scene_id: str = "two_g1_box_push_layout"
    payload_id: str = "long_box"
    robot_ids: tuple[str, ...] = ("front", "back")
    target_region: str = "manual_layout_target"
    task_stage: str = "full_scene"
    route_discovery: bool = False
    stage1_goal_x_m: float = 1.35
    stage1_goal_y_m: float = 2.35
    stage1_goal_tolerance_m: float = 0.20
    stage2_corridor_goal_x_m: float = 1.35
    stage2_corridor_goal_y_m: float = 7.60
    stage2_corridor_goal_tolerance_m: float = 0.25
    stage3_right_goal_x_m: float = -1.50
    stage3_right_goal_y_m: float = 7.60
    stage3_right_goal_tolerance_m: float = 0.25
    max_wall_contacts: int = 0
    min_payload_z_m: float = 0.80
    min_robot_pelvis_height_m: float = 0.35
    min_robot_xy_separation_m: float = 0.60
    require_payload_active: bool = True
    require_each_robot_payload_contact: bool = True
    min_forward_progress_m: float = 0.05
    require_payload_on_target: bool = True
    require_robots_exit: bool = False
    target_center_x_m: float = -1.5
    target_center_y_m: float = 7.6
    target_center_z_m: float = 0.3
    target_scale_x_m: float = 0.6
    target_scale_y_m: float = 0.6
    target_scale_z_m: float = 0.6
    target_xy_tolerance_m: float = DEFAULT_TARGET_XY_TOLERANCE_M
    target_z_tolerance_m: float = 0.08
    max_terminal_payload_tilt_deg: float = (
        DEFAULT_MAX_TERMINAL_PAYLOAD_TILT_DEG
    )
    require_terminal_payload_tilt_measurement: bool = False
    payload_half_height_m: float = 0.05
    robot_exit_min_distance_from_target_m: float = DEFAULT_ROBOT_EXIT_MIN_DISTANCE_M
    robot_pre_stand_clearance_margin_m: float = (
        DEFAULT_ROBOT_PRE_STAND_CLEARANCE_MARGIN_M
    )
    # Route semantics are task data. A different scene can provide a different
    # phase list without adding planner or verifier branches.
    mission_topology: tuple[str, ...] = (
        "transport through the initial straight passage",
        "negotiate the first corner into the cross passage",
        "transport through the cross passage",
        "negotiate the second corner into the final straight passage",
        "transport through the final straight passage",
        "place and release the payload on the target",
    )
    route_progress_gates: tuple[RouteProgressGate, ...] = (
        DEFAULT_MANUAL_LAYOUT_ROUTE_GATES
    )
    route_segments: tuple[RouteSegment, ...] = DEFAULT_MANUAL_LAYOUT_ROUTE_SEGMENTS


@dataclass(frozen=True)
class VisualObservation:
    """One external camera frame or text summary available to agents."""

    source: str
    frame_path: str = ""
    summary: str = ""
    captured_at_unix: float = 0.0
    used_by: tuple[str, ...] = ("task_orchestration", "collaborative_evolution_agent")


@dataclass(frozen=True)
class ObservationBundle:
    """Initial or rollout observation passed into orchestrator/evolution."""

    runtime_state_path: str = ""
    runtime_state: dict[str, Any] = field(default_factory=dict)
    visual_observations: tuple[VisualObservation, ...] = ()
    retrieved_specs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Subtask:
    """A task-level step that will be rendered into executable Python."""

    name: str
    description: str
    skill_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    guard: str = ""


@dataclass(frozen=True)
class RobotAssignment:
    """One robot's explicit role in a phase-aware collaborative policy."""

    robot_id: str
    role: str
    objective: str
    synchronization_group: str
    safety_constraints: tuple[str, ...] = ()
    phase_id: str = ""
    contact_intent: str = "unspecified"
    coordination_mode: CoordinationMode = "sync"


@dataclass(frozen=True)
class RobotPolicyScript:
    """Executable Python produced for one robot by the task orchestrator."""

    robot_id: str
    script_id: str
    code: str
    synchronization_group: str
    phase_id: str = ""
    contact_intent: str = "unspecified"
    coordination_mode: CoordinationMode = "sync"


@dataclass(frozen=True)
class PolicyScript:
    """Generated or repaired executable Python policy."""

    script_id: str
    attempt: int
    code: str
    plan: tuple[str, ...] = ()
    subtasks: tuple[Subtask, ...] = ()
    assignments: tuple[RobotAssignment, ...] = ()
    robot_scripts: tuple[RobotPolicyScript, ...] = ()
    parent_script_id: str = ""
    rationale: str = ""
    verification_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillCallRecord:
    """One Skill SDK call made by a policy script."""

    index: int
    skill_name: str
    parameters: dict[str, Any]
    status: str
    started_at_unix: float
    finished_at_unix: float
    artifact_path: str = ""
    stdout: str = ""
    stderr: str = ""
    robot_ids: tuple[str, ...] = ()
    effective_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedSkillCall:
    """One allowlisted call emitted by an individual robot script."""

    index: int
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotExecutionTrace:
    """Validated calls produced by one robot-specific policy script."""

    robot_id: str
    script_id: str
    status: str
    synchronization_group: str
    planned_calls: tuple[PlannedSkillCall, ...] = ()
    error: str = ""
    phase_id: str = ""
    contact_intent: str = "unspecified"
    coordination_mode: CoordinationMode = "sync"


@dataclass(frozen=True)
class ExecutionTrace:
    """Script execution trace used by the evaluator and memory updater."""

    script_id: str
    attempt: int
    status: str
    skill_calls: tuple[SkillCallRecord, ...] = ()
    robot_executions: tuple[RobotExecutionTrace, ...] = ()
    error: str = ""
    artifact_dir: str = ""
    verification_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    """Verifier output from physical/runtime state and optional VLM evidence."""

    success: bool
    failure_type: FailureType
    failed_subtask: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()
    visual_summary: str = ""
    failure_level: FailureLevel = "none"
    diagnosis: str = ""
    allowed_repairs: tuple[str, ...] = ()
    forbidden_repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvolutionRecord:
    """One iteration of script generation/execution/evaluation/evolution."""

    task: BoxCarryTask
    policy: PolicyScript
    observation: ObservationBundle
    execution: ExecutionTrace
    evaluation: EvaluationReport
    evolved_spec: str = ""
    next_script_id: str = ""
    checkpoint_before: str = ""
    checkpoint_after: str = ""
    rollback_performed: bool = False
    controller_decision: dict[str, Any] = field(default_factory=dict)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "__dataclass_fields__"):
        payload = asdict(payload)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, payloads: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            if hasattr(payload, "__dataclass_fields__"):
                payload = asdict(payload)
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
