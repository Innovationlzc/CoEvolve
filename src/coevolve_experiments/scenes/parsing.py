"""Strict parser for JSON-compatible scene registry documents."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from ..contracts import SceneId
from .schema import (
    CameraConfig,
    CorridorGeometry,
    CorridorSegment,
    Derivation,
    DirectedRouteSegment,
    EvidencePin,
    ExecutionEvidence,
    InitialState,
    Pose,
    ProgressGate,
    RealizationParameter,
    RobotInitialState,
    RouteConfig,
    SceneAsset,
    SceneSpec,
    TargetGeometry,
    Vec2,
    Vec3,
)


def scene_from_mapping(raw: Mapping[str, Any]) -> SceneSpec:
    """Strictly convert a decoded scene document to immutable records."""

    _require_keys(
        raw,
        {
            "schema_version",
            "scene_id",
            "label",
            "scene_hash",
            "protocol_invariant_ref",
            "terminal_predicate_ref",
            "asset",
            "initial_state",
            "target",
            "corridor",
            "camera",
            "route",
            "derivation",
            "evidence",
        },
        "scene",
    )

    asset_raw = _mapping(raw, "asset")
    initial_raw = _mapping(raw, "initial_state")
    target_raw = _mapping(raw, "target")
    corridor_raw = _mapping(raw, "corridor")
    camera_raw = _mapping(raw, "camera")
    route_raw = _mapping(raw, "route")
    derivation_raw = _mapping(raw, "derivation")
    evidence_raw = _mapping(raw, "evidence")
    _require_keys(
        asset_raw,
        {
            "source_path",
            "source_sha256",
            "realization",
            "realization_parameters",
            "execution_path",
            "execution_sha256",
            "execution_composite_sha256",
        },
        "asset",
    )
    _require_keys(
        initial_raw,
        {"checkpoint_id", "fresh_physical_state", "robots", "payload"},
        "initial_state",
    )
    _require_keys(target_raw, {"center", "radius_m"}, "target")
    _require_keys(
        corridor_raw, {"envelope_min", "envelope_max", "segments"}, "corridor"
    )
    _require_keys(
        camera_raw,
        {
            "global_eye",
            "global_target",
            "global_up",
            "payload_eye_offset",
            "payload_target_offset",
        },
        "camera",
    )
    _require_keys(
        route_raw,
        {
            "schema_version",
            "kind",
            "scene_id",
            "agent_visible",
            "host_validated_reference",
            "host_selects_primitive",
            "host_selects_coordination_mode",
            "host_selects_robot_roles",
            "host_selects_bounded_parameters",
            "mission_topology",
            "progress_gates",
            "segments",
        },
        "route",
    )
    _require_keys(
        derivation_raw, {"kind", "source_scene", "transform_axis"}, "derivation"
    )
    _require_keys(
        evidence_raw,
        {"simulator_integrity", "initial_collision", "scripted_solvability"},
        "evidence",
    )
    for index, value in enumerate(_sequence(asset_raw, "realization_parameters")):
        _require_keys(
            _mapping_value(value, "asset.realization_parameters"),
            {"parameter_id", "operation", "targets", "values"},
            f"asset.realization_parameters[{index}]",
        )
    for index, value in enumerate(_sequence(initial_raw, "robots")):
        robot = _mapping_value(value, "initial_state.robots")
        _require_keys(robot, {"robot_id", "pose"}, f"initial_state.robots[{index}]")
        _require_keys(
            _mapping(robot, "pose"),
            {"position", "yaw_rad"},
            f"initial_state.robots[{index}].pose",
        )
    _require_keys(_mapping(initial_raw, "payload"), {"position", "yaw_rad"}, "payload")
    for index, value in enumerate(_sequence(corridor_raw, "segments")):
        _require_keys(
            _mapping_value(value, "corridor.segments"),
            {"segment_id", "start", "end", "clear_width_m"},
            f"corridor.segments[{index}]",
        )
    for index, value in enumerate(_sequence(route_raw, "progress_gates")):
        _require_keys(
            _mapping_value(value, "route.progress_gates"),
            {"phase_name", "normal_x", "normal_y", "threshold_m"},
            f"route.progress_gates[{index}]",
        )
    for index, value in enumerate(_sequence(route_raw, "segments")):
        _require_keys(
            _mapping_value(value, "route.segments"),
            {
                "segment_id",
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
            },
            f"route.segments[{index}]",
        )

    execution_path = _optional_string(asset_raw, "execution_path")
    execution_sha256 = _optional_string(asset_raw, "execution_sha256")
    execution_composite_sha256 = _optional_string(asset_raw, "execution_composite_sha256")
    realization_parameters = tuple(
        RealizationParameter(
            parameter_id=_string(
                _mapping_value(value, "asset.realization_parameters"), "parameter_id"
            ),
            operation=_string(
                _mapping_value(value, "asset.realization_parameters"), "operation"
            ),
            targets=tuple(
                _string_value(item, "asset.realization_parameters.targets")
                for item in _sequence(
                    _mapping_value(value, "asset.realization_parameters"), "targets"
                )
            ),
            values=tuple(
                _number(item, "asset.realization_parameters.values")
                for item in _sequence(
                    _mapping_value(value, "asset.realization_parameters"), "values"
                )
            ),
        )
        for value in _sequence(asset_raw, "realization_parameters")
    )
    robots = tuple(
        RobotInitialState(
            robot_id=_string(_mapping_value(value, "initial_state.robots"), "robot_id"),
            pose=_pose(_mapping(_mapping_value(value, "initial_state.robots"), "pose")),
        )
        for value in _sequence(initial_raw, "robots")
    )
    segments = tuple(
        CorridorSegment(
            segment_id=_string(_mapping_value(value, "corridor.segments"), "segment_id"),
            start=_vec2(_required(_mapping_value(value, "corridor.segments"), "start"), "start"),
            end=_vec2(_required(_mapping_value(value, "corridor.segments"), "end"), "end"),
            clear_width_m=_number(
                _required(_mapping_value(value, "corridor.segments"), "clear_width_m"),
                "clear_width_m",
            ),
        )
        for value in _sequence(corridor_raw, "segments")
    )
    route_segments = tuple(
        DirectedRouteSegment(
            segment_id=_string(
                _mapping_value(value, "route.segments"), "segment_id"
            ),
            direction_x=_number(
                _required(_mapping_value(value, "route.segments"), "direction_x"),
                "route.segments.direction_x",
            ),
            direction_y=_number(
                _required(_mapping_value(value, "route.segments"), "direction_y"),
                "route.segments.direction_y",
            ),
            start_x_m=_number(
                _required(_mapping_value(value, "route.segments"), "start_x_m"),
                "route.segments.start_x_m",
            ),
            start_y_m=_number(
                _required(_mapping_value(value, "route.segments"), "start_y_m"),
                "route.segments.start_y_m",
            ),
            goal_x_m=_number(
                _required(_mapping_value(value, "route.segments"), "goal_x_m"),
                "route.segments.goal_x_m",
            ),
            goal_y_m=_number(
                _required(_mapping_value(value, "route.segments"), "goal_y_m"),
                "route.segments.goal_y_m",
            ),
            center_x_m=_number(
                _required(_mapping_value(value, "route.segments"), "center_x_m"),
                "route.segments.center_x_m",
            ),
            center_y_m=_number(
                _required(_mapping_value(value, "route.segments"), "center_y_m"),
                "route.segments.center_y_m",
            ),
            half_width_m=_number(
                _required(_mapping_value(value, "route.segments"), "half_width_m"),
                "route.segments.half_width_m",
            ),
            detour_margin_m=_number(
                _required(
                    _mapping_value(value, "route.segments"), "detour_margin_m"
                ),
                "route.segments.detour_margin_m",
            ),
        )
        for value in _sequence(route_raw, "segments")
    )
    progress_gates = tuple(
        ProgressGate(
            phase_name=_string(_mapping_value(value, "route.progress_gates"), "phase_name"),
            normal_x=_number(
                _required(_mapping_value(value, "route.progress_gates"), "normal_x"),
                "route.progress_gates.normal_x",
            ),
            normal_y=_number(
                _required(_mapping_value(value, "route.progress_gates"), "normal_y"),
                "route.progress_gates.normal_y",
            ),
            threshold_m=_number(
                _required(_mapping_value(value, "route.progress_gates"), "threshold_m"),
                "route.progress_gates.threshold_m",
            ),
        )
        for value in _sequence(route_raw, "progress_gates")
    )
    source_scene_value = derivation_raw.get("source_scene")
    if source_scene_value is not None and not isinstance(source_scene_value, str):
        raise ValueError("derivation.source_scene must be a string or null")
    transform_axis = _optional_string(derivation_raw, "transform_axis")
    return SceneSpec(
        schema_version=_integer(raw, "schema_version"),
        scene_id=SceneId(_string(raw, "scene_id")),
        label=_string(raw, "label"),
        scene_hash=_string(raw, "scene_hash"),
        protocol_invariant_ref=_string(raw, "protocol_invariant_ref"),
        terminal_predicate_ref=_string(raw, "terminal_predicate_ref"),
        asset=SceneAsset(
            source_path=_string(asset_raw, "source_path"),
            source_sha256=_string(asset_raw, "source_sha256"),
            realization=_string(asset_raw, "realization"),
            realization_parameters=realization_parameters,
            execution_path=execution_path,
            execution_sha256=execution_sha256,
            execution_composite_sha256=execution_composite_sha256,
        ),
        initial_state=InitialState(
            checkpoint_id=_string(initial_raw, "checkpoint_id"),
            fresh_physical_state=_boolean(initial_raw, "fresh_physical_state"),
            robots=robots,
            payload=_pose(_mapping(initial_raw, "payload")),
        ),
        target=TargetGeometry(
            center=_vec3(_required(target_raw, "center"), "target.center"),
            radius_m=_number(_required(target_raw, "radius_m"), "target.radius_m"),
        ),
        corridor=CorridorGeometry(
            envelope_min=_vec2(_required(corridor_raw, "envelope_min"), "corridor.envelope_min"),
            envelope_max=_vec2(_required(corridor_raw, "envelope_max"), "corridor.envelope_max"),
            segments=segments,
        ),
        camera=CameraConfig(
            global_eye=_vec3(_required(camera_raw, "global_eye"), "camera.global_eye"),
            global_target=_vec3(_required(camera_raw, "global_target"), "camera.global_target"),
            global_up=_vec3(_required(camera_raw, "global_up"), "camera.global_up"),
            payload_eye_offset=_vec3(
                _required(camera_raw, "payload_eye_offset"), "camera.payload_eye_offset"
            ),
            payload_target_offset=_vec3(
                _required(camera_raw, "payload_target_offset"), "camera.payload_target_offset"
            ),
        ),
        route=RouteConfig(
            schema_version=_integer(route_raw, "schema_version"),
            kind=_string(route_raw, "kind"),
            scene_id=_string(route_raw, "scene_id"),
            agent_visible=_boolean(route_raw, "agent_visible"),
            host_validated_reference=_boolean(
                route_raw, "host_validated_reference"
            ),
            host_selects_primitive=_boolean(route_raw, "host_selects_primitive"),
            host_selects_coordination_mode=_boolean(
                route_raw, "host_selects_coordination_mode"
            ),
            host_selects_robot_roles=_boolean(
                route_raw, "host_selects_robot_roles"
            ),
            host_selects_bounded_parameters=_boolean(
                route_raw, "host_selects_bounded_parameters"
            ),
            mission_topology=tuple(
                _string_value(value, "route.mission_topology")
                for value in _sequence(route_raw, "mission_topology")
            ),
            progress_gates=progress_gates,
            segments=route_segments,
        ),
        derivation=Derivation(
            kind=_string(derivation_raw, "kind"),
            source_scene=SceneId(source_scene_value) if source_scene_value is not None else None,
            transform_axis=transform_axis,
        ),
        evidence=ExecutionEvidence(
            simulator_integrity=_evidence_pin(evidence_raw.get("simulator_integrity")),
            initial_collision=_evidence_pin(evidence_raw.get("initial_collision")),
            scripted_solvability=_evidence_pin(evidence_raw.get("scripted_solvability")),
        ),
    )


def _evidence_pin(value: Any) -> Optional[EvidencePin]:
    if value is None:
        return None
    mapping = _mapping_value(value, "evidence pin")
    _require_keys(mapping, {"path", "sha256"}, "evidence pin")
    return EvidencePin(path=_string(mapping, "path"), sha256=_string(mapping, "sha256"))


def _pose(mapping: Mapping[str, Any]) -> Pose:
    _require_keys(mapping, {"position", "yaw_rad"}, "pose")
    return Pose(
        position=_vec3(_required(mapping, "position"), "pose.position"),
        yaw_rad=_number(_required(mapping, "yaw_rad"), "pose.yaw_rad"),
    )


def _vec2(value: Any, name: str) -> Vec2:
    items = _number_sequence(value, name, length=2)
    return Vec2(items[0], items[1])


def _vec3(value: Any, name: str) -> Vec3:
    items = _number_sequence(value, name, length=3)
    return Vec3(items[0], items[1], items[2])


def _number_sequence(value: Any, name: str, *, length: int) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be an array of length {length}")
    return tuple(_number(item, name) for item in value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must contain finite numbers")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = _required(mapping, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _boolean(mapping: Mapping[str, Any], key: str) -> bool:
    value = _required(mapping, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_value(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must contain non-empty strings")
    return value


def _optional_string(mapping: Mapping[str, Any], key: str) -> Optional[str]:
    value = mapping.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{key} must be a non-empty string or null")
    return value


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _mapping_value(_required(mapping, key), key)


def _mapping_value(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = _required(mapping, key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required key: {key}")
    return mapping[key]


def _require_keys(
    mapping: Mapping[str, Any], required: set[str], label: str
) -> None:
    actual = set(mapping)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing or unknown:
        raise ValueError(f"{label} fields differ: missing={missing}, unknown={unknown}")

__all__ = ["scene_from_mapping"]
