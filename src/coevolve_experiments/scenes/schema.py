"""Immutable scene-registry schema.

Terminal tolerances intentionally do not live here.  Scenes point to the
protocol lock's invariant and terminal fingerprints, which prevents a scene
variant from quietly changing success criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PurePosixPath
import re
from typing import Optional, Tuple

from ..contracts import SceneId


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __post_init__(self) -> None:
        _finite_number("Vec2.x", self.x)
        _finite_number("Vec2.y", self.y)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        _finite_number("Vec3.x", self.x)
        _finite_number("Vec3.y", self.y)
        _finite_number("Vec3.z", self.z)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Pose:
    position: Vec3
    yaw_rad: float

    def __post_init__(self) -> None:
        _finite_number("Pose.yaw_rad", self.yaw_rad)


@dataclass(frozen=True)
class RobotInitialState:
    robot_id: str
    pose: Pose


@dataclass(frozen=True)
class InitialState:
    checkpoint_id: str
    fresh_physical_state: bool
    robots: Tuple[RobotInitialState, ...]
    payload: Pose

    def __post_init__(self) -> None:
        if self.checkpoint_id != "c0" or not self.fresh_physical_state:
            raise ValueError("scene runs must begin from fresh physical state c0")
        ids = tuple(robot.robot_id for robot in self.robots)
        if ids != ("front", "back"):
            raise ValueError("initial state must contain immutable front/back robot IDs")


@dataclass(frozen=True)
class RealizationParameter:
    """One deterministic generator operation over selected authored prims."""

    parameter_id: str
    operation: str
    targets: Tuple[str, ...]
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.parameter_id or not self.operation:
            raise ValueError("realization parameter ID and operation cannot be empty")
        if not self.targets or any(not target for target in self.targets):
            raise ValueError("realization parameters require non-empty prim targets")
        for index, value in enumerate(self.values):
            _finite_number(f"realization parameter values[{index}]", value)
        value_counts = {
            "use_pinned_source_unchanged": 0,
            "apply_xy_matrix_and_pi_minus_yaw": 4,
            "set_box_transform_xyz_scale_xyz_yaw": 7,
            "remove_matching_prims": 0,
            "add_collision_box_xyz_scale_xyz_yaw": 7,
        }
        if self.operation not in value_counts:
            raise ValueError(f"unsupported realization operation: {self.operation}")
        expected_count = value_counts[self.operation]
        if len(self.values) != expected_count:
            raise ValueError(
                f"{self.operation} requires {expected_count} values, got {len(self.values)}"
            )
        if self.operation == "apply_xy_matrix_and_pi_minus_yaw" and self.values != (
            -1.0,
            0.0,
            0.0,
            1.0,
        ):
            raise ValueError("mirroring must use the frozen reflection matrix")
        if self.operation in {
            "set_box_transform_xyz_scale_xyz_yaw",
            "add_collision_box_xyz_scale_xyz_yaw",
        } and any(value <= 0 for value in self.values[3:6]):
            raise ValueError("collision-box realization scales must be positive")


@dataclass(frozen=True)
class SceneAsset:
    source_path: str
    source_sha256: str
    realization: str
    realization_parameters: Tuple[RealizationParameter, ...]
    execution_path: Optional[str]
    execution_sha256: Optional[str]
    execution_composite_sha256: Optional[str]

    def __post_init__(self) -> None:
        _relative_path("asset.source_path", self.source_path)
        _sha256("asset.source_sha256", self.source_sha256)
        if self.realization not in {
            "authored",
            "mirror_x",
            "narrow_corridor",
            "altered_topology",
        }:
            raise ValueError(f"unsupported asset realization: {self.realization}")
        if not self.realization_parameters:
            raise ValueError("asset realization requires deterministic generator parameters")
        parameter_ids = [item.parameter_id for item in self.realization_parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("asset realization parameter IDs must be unique")
        operations = tuple(item.operation for item in self.realization_parameters)
        if self.realization == "authored" and operations != (
            "use_pinned_source_unchanged",
        ):
            raise ValueError("authored asset must use its pinned source unchanged")
        if self.realization == "mirror_x" and operations != (
            "apply_xy_matrix_and_pi_minus_yaw",
        ):
            raise ValueError("mirrored asset requires exactly one frozen reflection")
        if self.realization == "narrow_corridor" and set(operations) != {
            "set_box_transform_xyz_scale_xyz_yaw"
        }:
            raise ValueError("narrow asset may contain only explicit wall transforms")
        if self.realization == "altered_topology" and (
            operations[0] != "remove_matching_prims"
            or set(operations[1:]) != {"add_collision_box_xyz_scale_xyz_yaw"}
        ):
            raise ValueError(
                "altered topology must remove authored walls then add explicit collision boxes"
            )
        declared_execution_fields = (
            self.execution_path,
            self.execution_sha256,
            self.execution_composite_sha256,
        )
        if any(value is None for value in declared_execution_fields) and any(
            value is not None for value in declared_execution_fields
        ):
            raise ValueError(
                "execution_path, execution_sha256, and execution_composite_sha256 "
                "must be declared together"
            )
        if self.execution_path is not None:
            _relative_path("asset.execution_path", self.execution_path)
            _sha256("asset.execution_sha256", self.execution_sha256 or "")
            _sha256(
                "asset.execution_composite_sha256",
                self.execution_composite_sha256 or "",
            )


@dataclass(frozen=True)
class TargetGeometry:
    center: Vec3
    radius_m: float

    def __post_init__(self) -> None:
        _finite_number("target radius_m", self.radius_m)
        if self.radius_m <= 0:
            raise ValueError("target radius must be positive")


@dataclass(frozen=True)
class CorridorSegment:
    segment_id: str
    start: Vec2
    end: Vec2
    clear_width_m: float

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("corridor segment ID cannot be empty")
        _finite_number("corridor clear_width_m", self.clear_width_m)
        if self.start == self.end:
            raise ValueError(f"corridor segment {self.segment_id} has zero length")
        if self.clear_width_m <= 0:
            raise ValueError("corridor clear width must be positive")


@dataclass(frozen=True)
class CorridorGeometry:
    envelope_min: Vec2
    envelope_max: Vec2
    segments: Tuple[CorridorSegment, ...]

    def __post_init__(self) -> None:
        if self.envelope_min.x >= self.envelope_max.x or self.envelope_min.y >= self.envelope_max.y:
            raise ValueError("corridor envelope min must be below max")
        if not self.segments:
            raise ValueError("corridor must contain at least one segment")
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("corridor segment IDs must be unique")


@dataclass(frozen=True)
class CameraConfig:
    global_eye: Vec3
    global_target: Vec3
    global_up: Vec3
    payload_eye_offset: Vec3
    payload_target_offset: Vec3

    def __post_init__(self) -> None:
        if self.global_eye == self.global_target:
            raise ValueError("global camera eye and target cannot coincide")
        if self.global_up == Vec3(0.0, 0.0, 0.0):
            raise ValueError("global camera up vector cannot be zero")


@dataclass(frozen=True)
class ProgressGate:
    phase_name: str
    normal_x: float
    normal_y: float
    threshold_m: float

    def __post_init__(self) -> None:
        if not self.phase_name:
            raise ValueError("progress gate phase_name cannot be empty")
        _finite_number("progress gate normal_x", self.normal_x)
        _finite_number("progress gate normal_y", self.normal_y)
        _finite_number("progress gate threshold_m", self.threshold_m)
        norm_sq = self.normal_x * self.normal_x + self.normal_y * self.normal_y
        if abs(norm_sq - 1.0) > 1e-9:
            raise ValueError("progress gate normal must be unit length")


@dataclass(frozen=True)
class DirectedRouteSegment:
    """One V83-compatible connected-route segment.

    Field names deliberately match ``praxis_coop.cap_policy.RouteSegment`` so
    the registry can emit the exact V83 route document without reconstructing
    directions, centers, or clearance widths from a lossy polyline.
    """

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
    detour_margin_m: float

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("directed route segment ID cannot be empty")
        for name in (
            "direction_x",
            "direction_y",
            "start_x_m",
            "start_y_m",
            "goal_x_m",
            "goal_y_m",
            "center_x_m",
            "center_y_m",
            "half_width_m",
            "detour_margin_m",
        ):
            _finite_number(f"directed route segment {name}", getattr(self, name))
        direction_norm_sq = (
            self.direction_x * self.direction_x + self.direction_y * self.direction_y
        )
        if abs(direction_norm_sq - 1.0) > 1e-9:
            raise ValueError("directed route segment direction must be unit length")
        if self.half_width_m <= 0.0:
            raise ValueError("directed route segment half_width_m must be positive")
        if self.detour_margin_m < 0.0:
            raise ValueError(
                "directed route segment detour_margin_m must be non-negative"
            )
        along_length = (
            (self.goal_x_m - self.start_x_m) * self.direction_x
            + (self.goal_y_m - self.start_y_m) * self.direction_y
        )
        if along_length <= 1.0e-9:
            raise ValueError(
                "directed route segment goal must be ahead of start along direction"
            )
        center_projection = (
            self.center_x_m * self.direction_x
            + self.center_y_m * self.direction_y
        )
        start_projection = (
            self.start_x_m * self.direction_x
            + self.start_y_m * self.direction_y
        )
        goal_projection = (
            self.goal_x_m * self.direction_x
            + self.goal_y_m * self.direction_y
        )
        if not (
            min(start_projection, goal_projection) - 1.0e-9
            <= center_projection
            <= max(start_projection, goal_projection) + 1.0e-9
        ):
            raise ValueError(
                "directed route segment center must lie between start and goal"
            )


@dataclass(frozen=True)
class RouteConfig:
    schema_version: int
    kind: str
    scene_id: str
    agent_visible: bool
    host_validated_reference: bool
    host_selects_primitive: bool
    host_selects_coordination_mode: bool
    host_selects_robot_roles: bool
    host_selects_bounded_parameters: bool
    mission_topology: Tuple[str, ...]
    progress_gates: Tuple[ProgressGate, ...]
    segments: Tuple[DirectedRouteSegment, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("directed route schema_version must equal exact V83 version 1")
        if self.kind != "directed_connected_route":
            raise ValueError("route kind must be 'directed_connected_route'")
        if not self.scene_id:
            raise ValueError("directed route scene_id cannot be empty")
        if not self.agent_visible:
            raise ValueError(
                "the exact V83 Host-validated route reference is model-visible"
            )
        if not self.host_validated_reference:
            raise ValueError("route reference must be Host-validated")
        if any(
            (
                self.host_selects_primitive,
                self.host_selects_coordination_mode,
                self.host_selects_robot_roles,
                self.host_selects_bounded_parameters,
            )
        ):
            raise ValueError(
                "route Host may not select primitives, coordination, robot roles, "
                "or bounded parameters"
            )
        if not self.mission_topology or any(not phase for phase in self.mission_topology):
            raise ValueError("route mission_topology cannot be empty")
        if not self.progress_gates:
            raise ValueError("route must contain oriented progress gates")
        phase_names = [gate.phase_name for gate in self.progress_gates]
        if len(phase_names) != len(set(phase_names)):
            raise ValueError("progress gate phase names must be unique")
        if len(self.segments) != len(self.progress_gates) + 1:
            raise ValueError(
                "directed route segment count must equal progress-gate count + 1"
            )
        if len(self.mission_topology) != len(self.segments) + 1:
            raise ValueError(
                "route mission_topology must contain one phase per segment plus terminal"
            )
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("directed route segment IDs must be unique")
        for index, (incoming, outgoing) in enumerate(
            zip(self.segments, self.segments[1:])
        ):
            gap = math.hypot(
                incoming.goal_x_m - outgoing.start_x_m,
                incoming.goal_y_m - outgoing.start_y_m,
            )
            if gap > 1.0e-9:
                raise ValueError(
                    f"directed route segments {index} and {index + 1} are disconnected"
                )
        for index, gate in enumerate(self.progress_gates):
            incoming = self.segments[index]
            direction_alignment = (
                gate.normal_x * incoming.direction_x
                + gate.normal_y * incoming.direction_y
            )
            if direction_alignment < 0.85:
                raise ValueError(
                    f"progress gate {gate.phase_name} must face along incoming segment"
                )
            start_projection = (
                gate.normal_x * incoming.start_x_m
                + gate.normal_y * incoming.start_y_m
            )
            goal_projection = (
                gate.normal_x * incoming.goal_x_m
                + gate.normal_y * incoming.goal_y_m
            )
            if start_projection > gate.threshold_m + 1.0e-9:
                raise ValueError(
                    f"progress gate {gate.phase_name} lies behind incoming start"
                )
            if goal_projection < gate.threshold_m - 1.0e-9:
                raise ValueError(
                    f"incoming segment does not reach progress gate {gate.phase_name}"
                )


@dataclass(frozen=True)
class Derivation:
    kind: str
    source_scene: Optional[SceneId]
    transform_axis: Optional[str]

    def __post_init__(self) -> None:
        if self.kind not in {"authored", "mirrored", "narrow", "altered_topology"}:
            raise ValueError(f"unsupported scene derivation: {self.kind}")
        if self.kind == "authored":
            if self.source_scene is not None or self.transform_axis is not None:
                raise ValueError("authored scene cannot declare a source transform")
        elif self.source_scene is not SceneId.BASE:
            raise ValueError("candidate scene variants must derive from the base scene")
        if self.kind == "mirrored" and self.transform_axis != "x=0":
            raise ValueError("the mirrored scene must reflect across x=0")
        if self.kind != "mirrored" and self.transform_axis is not None:
            raise ValueError("only the mirrored scene declares a transform axis")


@dataclass(frozen=True)
class EvidencePin:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _relative_path("evidence path", self.path)
        _sha256("evidence sha256", self.sha256)


@dataclass(frozen=True)
class ExecutionEvidence:
    simulator_integrity: Optional[EvidencePin]
    initial_collision: Optional[EvidencePin]
    scripted_solvability: Optional[EvidencePin]

    def missing_names(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name in (
                "simulator_integrity",
                "initial_collision",
                "scripted_solvability",
            )
            if getattr(self, name) is None
        )


@dataclass(frozen=True)
class SceneSpec:
    schema_version: int
    scene_id: SceneId
    label: str
    scene_hash: str
    protocol_invariant_ref: str
    terminal_predicate_ref: str
    asset: SceneAsset
    initial_state: InitialState
    target: TargetGeometry
    corridor: CorridorGeometry
    camera: CameraConfig
    route: RouteConfig
    derivation: Derivation
    evidence: ExecutionEvidence

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported scene schema {self.schema_version}")
        if not self.label:
            raise ValueError("scene label cannot be empty")
        _sha256("scene_hash", self.scene_hash)
        if self.protocol_invariant_ref != "lock://invariant_fingerprint":
            raise ValueError("scene must reference the protocol invariant lock")
        if self.terminal_predicate_ref != "lock://terminal_fingerprint":
            raise ValueError("scene must reference the terminal predicate lock")
        expected_kind = {
            SceneId.BASE: "authored",
            SceneId.MIRRORED: "mirrored",
            SceneId.NARROW: "narrow",
            SceneId.ALTERED_TOPOLOGY: "altered_topology",
        }[self.scene_id]
        if self.derivation.kind != expected_kind:
            raise ValueError(f"scene {self.scene_id.value} has the wrong derivation kind")


def _sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")


def _finite_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")


def _relative_path(name: str, value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{name} must be a normalized POSIX relative path")


__all__ = [
    "CameraConfig",
    "CorridorGeometry",
    "CorridorSegment",
    "Derivation",
    "DirectedRouteSegment",
    "EvidencePin",
    "ExecutionEvidence",
    "InitialState",
    "Pose",
    "ProgressGate",
    "RealizationParameter",
    "RobotInitialState",
    "RouteConfig",
    "SceneAsset",
    "SceneSpec",
    "TargetGeometry",
    "Vec2",
    "Vec3",
]
