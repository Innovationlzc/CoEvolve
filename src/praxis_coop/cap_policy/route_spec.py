"""Data-driven route geometry for scene-independent task guidance.

The route specification describes connected free-space segments and their
completion predicates.  It never prescribes a robot primitive, role mapping,
coordination mode, or action magnitude.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from math import hypot, isfinite
from pathlib import Path
from typing import Any

from .contracts import BoxCarryTask, RouteProgressGate, RouteSegment


ROUTE_SPEC_SCHEMA_VERSION = 1
ROUTE_REFERENCE_MIN_DOT = 0.85
ROUTE_GEOMETRY_TOLERANCE_M = 1.0e-6
_ROUTE_MOTION_METHODS = frozenset(
    {
        "coordinated_linear_transport",
        "coordinated_arc_transport",
        "coordinated_pivot",
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
        "turn_clockwise",
        "turn_counterclockwise",
    }
)


class RouteSpecConfigurationError(ValueError):
    """Raised before execution when task-authored route geometry is invalid."""


def load_task_route_spec(task: BoxCarryTask, path: str | Path) -> BoxCarryTask:
    """Load one scene route document and return a validated task copy.

    A route document owns only connected free-space geometry and semantic labels.
    The task continues to own its physical target contract, while the Agent owns
    every executable primitive, role assignment, synchronization relation, and
    bounded parameter choice.
    """

    route_path = Path(path).expanduser()
    try:
        payload = json.loads(route_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RouteSpecConfigurationError(
            f"cannot read route spec {route_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RouteSpecConfigurationError(
            f"route spec {route_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RouteSpecConfigurationError(
            f"route spec {route_path} must contain a JSON object"
        )
    try:
        schema_version = int(payload.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise RouteSpecConfigurationError(
            f"route spec {route_path} has an invalid schema_version"
        ) from exc
    if schema_version != ROUTE_SPEC_SCHEMA_VERSION:
        raise RouteSpecConfigurationError(
            f"route spec {route_path} requires schema_version "
            f"{ROUTE_SPEC_SCHEMA_VERSION}; got {schema_version}"
        )
    allowed_keys = {
        "schema_version",
        "kind",
        "scene_id",
        "mission_topology",
        "progress_gates",
        "segments",
    }
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise RouteSpecConfigurationError(
            f"route spec {route_path} contains unsupported fields {unknown_keys}; "
            "route data cannot prescribe actions, roles, synchronization, or parameters"
        )
    kind = str(payload.get("kind", "")).strip()
    if kind != "directed_connected_route":
        raise RouteSpecConfigurationError(
            f"route spec {route_path} kind must be 'directed_connected_route'; "
            f"got {kind!r}"
        )
    scene_id = str(payload.get("scene_id", "")).strip()
    if scene_id and scene_id != task.scene_id:
        raise RouteSpecConfigurationError(
            f"route spec scene_id {scene_id!r} does not match task scene_id "
            f"{task.scene_id!r}"
        )
    raw_gates = payload.get("progress_gates", ())
    raw_segments = payload.get("segments", ())
    raw_topology = payload.get("mission_topology", task.mission_topology)
    if not isinstance(raw_gates, list) or not isinstance(raw_segments, list):
        raise RouteSpecConfigurationError(
            f"route spec {route_path} requires progress_gates and segments arrays"
        )
    if not isinstance(raw_topology, (list, tuple)):
        raise RouteSpecConfigurationError(
            f"route spec {route_path} mission_topology must be an array"
        )
    try:
        gates = tuple(RouteProgressGate(**item) for item in raw_gates)
        segments = tuple(RouteSegment(**item) for item in raw_segments)
    except (TypeError, ValueError) as exc:
        raise RouteSpecConfigurationError(
            f"route spec {route_path} does not match the route schema: {exc}"
        ) from exc
    configured = replace(
        task,
        mission_topology=tuple(str(item).strip() for item in raw_topology),
        route_progress_gates=gates,
        route_segments=segments,
    )
    validate_task_route_spec(configured, source=str(route_path))
    return configured


def validate_task_route_spec(
    task: BoxCarryTask,
    *,
    source: str = "BoxCarryTask",
) -> dict[str, Any]:
    """Validate a connected route before any model call or physical dispatch."""

    failures: list[str] = []
    gates = tuple(task.route_progress_gates)
    segments = tuple(task.route_segments)
    if len(segments) != len(gates) + 1:
        failures.append(
            "segment count must equal progress-gate count + 1 "
            f"({len(segments)} != {len(gates)} + 1)"
        )
    if len(task.mission_topology) != len(segments) + 1:
        failures.append(
            "mission_topology must contain one entry per transport segment plus "
            f"one terminal placement entry ({len(task.mission_topology)} != "
            f"{len(segments)} + 1)"
        )

    segment_ids: set[str] = set()
    for index, segment in enumerate(segments):
        label = f"segments[{index}]"
        segment_id = str(segment.segment_id).strip()
        if not segment_id:
            failures.append(f"{label}.segment_id is empty")
        elif segment_id in segment_ids:
            failures.append(f"{label}.segment_id {segment_id!r} is duplicated")
        segment_ids.add(segment_id)
        values = (
            segment.direction_x,
            segment.direction_y,
            segment.start_x_m,
            segment.start_y_m,
            segment.goal_x_m,
            segment.goal_y_m,
            segment.center_x_m,
            segment.center_y_m,
            segment.half_width_m,
            segment.detour_margin_m,
        )
        if not all(isfinite(float(value)) for value in values):
            failures.append(f"{label} contains a non-finite number")
            continue
        direction_norm = hypot(segment.direction_x, segment.direction_y)
        if abs(direction_norm - 1.0) > 1.0e-6:
            failures.append(
                f"{label} direction must be normalized; norm={direction_norm:.9g}"
            )
        if segment.half_width_m <= 0.0:
            failures.append(f"{label}.half_width_m must be positive")
        if segment.detour_margin_m < 0.0:
            failures.append(f"{label}.detour_margin_m must be non-negative")
        if direction_norm > 1.0e-9:
            along_length = (
                (segment.goal_x_m - segment.start_x_m) * segment.direction_x
                + (segment.goal_y_m - segment.start_y_m) * segment.direction_y
            ) / direction_norm
            if along_length <= ROUTE_GEOMETRY_TOLERANCE_M:
                failures.append(
                    f"{label} goal must be ahead of start along its direction"
                )
            start_cross = _point_cross_track_error(
                segment,
                (segment.start_x_m, segment.start_y_m),
            )
            goal_cross = _point_cross_track_error(
                segment,
                (segment.goal_x_m, segment.goal_y_m),
            )
            if max(start_cross, goal_cross) > (
                segment.half_width_m + ROUTE_GEOMETRY_TOLERANCE_M
            ):
                failures.append(
                    f"{label} start/goal are outside its centered corridor width"
                )
            center_projection = (
                segment.center_x_m * segment.direction_x
                + segment.center_y_m * segment.direction_y
            )
            start_projection = (
                segment.start_x_m * segment.direction_x
                + segment.start_y_m * segment.direction_y
            )
            goal_projection = (
                segment.goal_x_m * segment.direction_x
                + segment.goal_y_m * segment.direction_y
            )
            if not (
                min(start_projection, goal_projection)
                - ROUTE_GEOMETRY_TOLERANCE_M
                <= center_projection
                <= max(start_projection, goal_projection)
                + ROUTE_GEOMETRY_TOLERANCE_M
            ):
                failures.append(
                    f"{label}.center must lie between start and goal along the segment"
                )

    gate_names: set[str] = set()
    for index, gate in enumerate(gates):
        label = f"progress_gates[{index}]"
        phase_name = str(gate.phase_name).strip()
        if not phase_name:
            failures.append(f"{label}.phase_name is empty")
        elif phase_name in gate_names:
            failures.append(f"{label}.phase_name {phase_name!r} is duplicated")
        gate_names.add(phase_name)
        values = (gate.normal_x, gate.normal_y, gate.threshold_m)
        if not all(isfinite(float(value)) for value in values):
            failures.append(f"{label} contains a non-finite number")
            continue
        normal_norm = hypot(gate.normal_x, gate.normal_y)
        if abs(normal_norm - 1.0) > 1.0e-6:
            failures.append(
                f"{label} normal must be normalized; norm={normal_norm:.9g}"
            )
            continue
        if index >= len(segments):
            continue
        incoming = segments[index]
        incoming_direction_dot = (
            gate.normal_x * incoming.direction_x
            + gate.normal_y * incoming.direction_y
        )
        if incoming_direction_dot < ROUTE_REFERENCE_MIN_DOT:
            failures.append(
                f"{label} normal must face along the incoming segment direction"
            )
        start_projection = (
            gate.normal_x * incoming.start_x_m
            + gate.normal_y * incoming.start_y_m
        )
        goal_projection = (
            gate.normal_x * incoming.goal_x_m
            + gate.normal_y * incoming.goal_y_m
        )
        if start_projection > gate.threshold_m + ROUTE_GEOMETRY_TOLERANCE_M:
            failures.append(
                f"{label} lies behind the incoming segment start"
            )
        if goal_projection < gate.threshold_m - ROUTE_GEOMETRY_TOLERANCE_M:
            failures.append(
                f"segments[{index}] does not reach {label}"
            )
        if index + 1 < len(segments):
            outgoing = segments[index + 1]
            gate_point = _segment_gate_intersection(incoming, gate)
            if gate_point is None or not _point_in_segment_envelope(
                outgoing,
                gate_point,
            ):
                failures.append(
                    f"segments[{index + 1}] does not cover the boundary of {label}"
                )

    for index, (incoming, outgoing) in enumerate(zip(segments, segments[1:])):
        connection_gap = hypot(
            incoming.goal_x_m - outgoing.start_x_m,
            incoming.goal_y_m - outgoing.start_y_m,
        )
        if connection_gap > ROUTE_GEOMETRY_TOLERANCE_M:
            failures.append(
                f"segments[{index}] and segments[{index + 1}] are disconnected; "
                f"endpoint gap={connection_gap:.6g}m"
            )

    final_target_covered = False
    if segments:
        final = segments[-1]
        direction = _unit(final.direction_x, final.direction_y)
        if direction is not None:
            target_rel_x = task.target_center_x_m - final.center_x_m
            target_rel_y = task.target_center_y_m - final.center_y_m
            target_cross = abs(
                -target_rel_x * direction[1] + target_rel_y * direction[0]
            )
            start_projection = (
                final.start_x_m * direction[0] + final.start_y_m * direction[1]
            )
            goal_projection = (
                final.goal_x_m * direction[0] + final.goal_y_m * direction[1]
            )
            target_projection = (
                task.target_center_x_m * direction[0]
                + task.target_center_y_m * direction[1]
            )
            final_target_covered = bool(
                target_cross <= final.half_width_m + task.target_xy_tolerance_m
                and min(start_projection, goal_projection)
                - task.target_xy_tolerance_m
                <= target_projection
                <= max(start_projection, goal_projection)
                + task.target_xy_tolerance_m
                and hypot(
                    final.goal_x_m - task.target_center_x_m,
                    final.goal_y_m - task.target_center_y_m,
                )
                <= task.target_xy_tolerance_m + ROUTE_GEOMETRY_TOLERANCE_M
            )
        if not final_target_covered:
            failures.append(
                "final segment must terminate inside and geometrically cover the "
                "task target region"
            )

    if failures:
        raise RouteSpecConfigurationError(
            f"invalid route specification from {source}: " + "; ".join(failures)
        )
    return {
        "valid": True,
        "source": str(source),
        "schema_version": ROUTE_SPEC_SCHEMA_VERSION,
        "segment_count": len(segments),
        "progress_gate_count": len(gates),
        "terminal_target_covered": final_target_covered,
        "agent_action_fields_present": False,
    }


def route_spec_document(task: BoxCarryTask) -> dict[str, Any]:
    """Return the task-authored route as model-readable geometry data."""

    return {
        "schema_version": ROUTE_SPEC_SCHEMA_VERSION,
        "kind": "directed_connected_route",
        "scene_id": task.scene_id,
        "mission_topology": list(task.mission_topology),
        "progress_gates": [asdict(gate) for gate in task.route_progress_gates],
        "segments": [asdict(segment) for segment in task.route_segments],
        "terminal_region": {
            "center_xy_m": [task.target_center_x_m, task.target_center_y_m],
            "xy_tolerance_m": task.target_xy_tolerance_m,
        },
        "agent_control": {
            "selects_primitive": True,
            "selects_coordination_mode": True,
            "selects_robot_roles": True,
            "selects_bounded_parameters": True,
        },
        "host_scope": (
            "The host validates route-reference geometry and completion predicates; "
            "it does not provide an action sequence."
        ),
    }


def active_route_segment(
    task: BoxCarryTask,
    *,
    completed_prefix: int,
) -> RouteSegment | None:
    """Resolve the current segment from a monotonic completed-gate prefix."""

    if not task.route_segments:
        return None
    index = max(0, int(completed_prefix))
    if index >= len(task.route_segments):
        return None
    return task.route_segments[index]


def route_guidance(
    task: BoxCarryTask,
    runtime_state: dict[str, Any] | None,
    *,
    completed_prefix: int,
) -> dict[str, Any]:
    """Build current-segment guidance without selecting an executable action."""

    segment = active_route_segment(task, completed_prefix=completed_prefix)
    if segment is None:
        return {}
    payload_xy = _payload_xy(runtime_state)
    along_progress = None
    cross_track_error = None
    if payload_xy is not None:
        direction = _unit(segment.direction_x, segment.direction_y)
        if direction is not None:
            rel_x = payload_xy[0] - segment.center_x_m
            rel_y = payload_xy[1] - segment.center_y_m
            along_progress = rel_x * direction[0] + rel_y * direction[1]
            cross_track_error = abs(-rel_x * direction[1] + rel_y * direction[0])
    segment_index = max(0, int(completed_prefix))
    return {
        "schema_version": ROUTE_SPEC_SCHEMA_VERSION,
        "active": True,
        "segment_index": segment_index,
        "segment_id": segment.segment_id,
        "world_direction_xy": [segment.direction_x, segment.direction_y],
        "local_goal_xy_m": [segment.goal_x_m, segment.goal_y_m],
        "corridor": {
            "start_xy_m": [segment.start_x_m, segment.start_y_m],
            "end_xy_m": [segment.goal_x_m, segment.goal_y_m],
            "center_xy_m": [segment.center_x_m, segment.center_y_m],
            "half_width_m": segment.half_width_m,
            "detour_margin_m": segment.detour_margin_m,
        },
        "measured": {
            "payload_xy_m": list(payload_xy) if payload_xy is not None else [],
            "along_segment_progress_m": _rounded(along_progress),
            "cross_track_error_m": _rounded(cross_track_error),
        },
        "completion_gate": _completion_gate(task, completed_prefix),
        "contract": (
            "Choose the coordination primitive, sync/async relation, roles, and "
            "bounded parameters yourself. Preserve this validated route reference "
            "until its completion gate is crossed or the turn budget requests a "
            "fresh route declaration."
        ),
    }


def validate_declared_route_reference(
    task: BoxCarryTask,
    runtime_state: dict[str, Any] | None,
    *,
    completed_prefix: int,
    declared_world_direction_xy: object,
) -> dict[str, Any]:
    """Validate a model-declared direction against the active connected segment."""

    guidance = route_guidance(
        task,
        runtime_state,
        completed_prefix=completed_prefix,
    )
    expected = _unit_pair(guidance.get("world_direction_xy"))
    declared = _unit_pair(declared_world_direction_xy)
    if expected is None or declared is None:
        return {
            "valid": False,
            "reason": "active route direction or declared world direction is unavailable",
            "guidance": guidance,
        }
    dot = expected[0] * declared[0] + expected[1] * declared[1]
    return {
        "valid": dot >= ROUTE_REFERENCE_MIN_DOT,
        "direction_dot": round(dot, 6),
        "minimum_direction_dot": ROUTE_REFERENCE_MIN_DOT,
        "declared_world_direction_xy": [round(declared[0], 6), round(declared[1], 6)],
        "validated_world_direction_xy": [round(expected[0], 6), round(expected[1], 6)],
        "reason": (
            "declared direction follows the active connected route segment"
            if dot >= ROUTE_REFERENCE_MIN_DOT
            else "declared direction contradicts the active connected route segment"
        ),
        "guidance": guidance,
    }


def validate_task_plan_route_reference(
    task: BoxCarryTask,
    runtime_state: dict[str, Any] | None,
    *,
    completed_prefix: int,
    task_plan: dict[str, Any],
) -> dict[str, Any]:
    """Validate and freeze the route reference before policy generation.

    Route geometry is task data. The model still owns the executable primitive,
    coordination mode, roles, and parameters, but a mutation cannot rewrite the
    connected segment that the current subtask is meant to traverse.
    """

    guidance = route_guidance(
        task,
        runtime_state,
        completed_prefix=completed_prefix,
    )
    if not guidance:
        return {"active": False, "reason": "no active transport segment"}
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    strategy = subtask.get("strategy", {})
    strategy = strategy if isinstance(strategy, dict) else {}
    method = str(strategy.get("primitive", "")).strip()
    navigation = subtask.get("navigation", {})
    navigation = navigation if isinstance(navigation, dict) else {}
    reference = str(navigation.get("reference", "")).strip()

    if method not in _ROUTE_MOTION_METHODS:
        return {
            "active": False,
            "reason": "current subtask is formation or payload-lifecycle setup",
            "guidance": guidance,
        }
    if reference != "visible_connected_free_space":
        return {
            "active": False,
            "reason": (
                "route-segment validation applies only to "
                "visible_connected_free_space; point-target references are "
                "validated by local navigation"
            ),
            "reference": reference,
            "guidance": guidance,
        }
    fresh_observation = _fresh_route_observation_after_turn_budget(
        runtime_state,
    )
    if fresh_observation.get("required", False) and not fresh_observation.get(
        "valid", False
    ):
        return {
            "active": True,
            "valid": False,
            "reason": str(
                fresh_observation.get(
                    "reason",
                    "turn-budget replan requires a fresh global observation",
                )
            ),
            "guidance": guidance,
            "fresh_global_observation": fresh_observation,
        }
    validation = validate_declared_route_reference(
        task,
        runtime_state,
        completed_prefix=completed_prefix,
        declared_world_direction_xy=navigation.get("world_direction_xy"),
    )
    if not validation.get("valid", False):
        return {"active": True, **validation}

    declared = list(validation["declared_world_direction_xy"])
    validated = list(validation["validated_world_direction_xy"])
    navigation["declared_world_direction_xy"] = declared
    navigation["world_direction_xy"] = validated
    navigation["completion_world_direction_xy"] = validated
    navigation["validated_route_segment_id"] = str(guidance["segment_id"])
    navigation["route_reference_locked"] = True
    observation = (
        runtime_state.get("global_observation", {})
        if isinstance(runtime_state, dict)
        else {}
    )
    observation = observation if isinstance(observation, dict) else {}
    try:
        route_observation_captured_at = float(
            observation.get("captured_at_unix", 0.0)
        )
    except (TypeError, ValueError):
        route_observation_captured_at = 0.0
    if route_observation_captured_at > 0.0:
        navigation["route_observation_captured_at_unix"] = (
            route_observation_captured_at
        )
    if fresh_observation.get("required", False):
        navigation["turn_budget_replan_validated"] = True
        navigation["route_observation_captured_at_unix"] = fresh_observation.get(
            "captured_at_unix"
        )
    subtask["navigation"] = navigation
    task_plan["current_subtask"] = subtask
    return {
        "active": True,
        **validation,
        "route_reference_locked": True,
        "fresh_global_observation": fresh_observation,
    }


def validate_locked_route_reference_unchanged(
    previous_task_plan: dict[str, Any] | None,
    candidate_task_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep a validated route segment immutable across one repair mutation."""

    previous = _task_plan_navigation(previous_task_plan)
    if not bool(previous.get("route_reference_locked", False)):
        return {"active": False, "reason": "no previously locked route reference"}
    candidate = _task_plan_navigation(candidate_task_plan)
    expected_reference = str(previous.get("reference", "")).strip()
    actual_reference = str(candidate.get("reference", "")).strip()
    expected_segment = str(previous.get("validated_route_segment_id", "")).strip()
    actual_segment = str(candidate.get("validated_route_segment_id", "")).strip()
    expected_direction = _unit_pair(previous.get("world_direction_xy"))
    actual_direction = _unit_pair(candidate.get("world_direction_xy"))
    direction_matches = bool(
        expected_direction is not None
        and actual_direction is not None
        and expected_direction[0] * actual_direction[0]
        + expected_direction[1] * actual_direction[1]
        >= 1.0 - 1.0e-6
    )
    valid = bool(
        candidate.get("route_reference_locked", False)
        and expected_reference == actual_reference
        and expected_segment == actual_segment
        and direction_matches
    )
    return {
        "active": True,
        "valid": valid,
        "expected_reference": expected_reference,
        "candidate_reference": actual_reference,
        "expected_segment_id": expected_segment,
        "candidate_segment_id": actual_segment,
        "expected_world_direction_xy": list(expected_direction or ()),
        "candidate_world_direction_xy": list(actual_direction or ()),
        "reason": (
            "validated route reference is unchanged"
            if valid
            else "repair mutation changed or removed the validated route reference"
        ),
    }


def route_segment_state(
    task: BoxCarryTask,
    runtime_state: dict[str, Any] | None,
    *,
    completed_prefix: int,
) -> dict[str, Any]:
    """Measure corridor membership and local-goal progress for one state."""

    guidance = route_guidance(
        task,
        runtime_state,
        completed_prefix=completed_prefix,
    )
    payload_xy = _payload_xy(runtime_state)
    if not guidance or payload_xy is None:
        return {"available": False}
    direction = _unit_pair(guidance.get("world_direction_xy"))
    corridor = guidance.get("corridor", {})
    center = corridor.get("center_xy_m", ()) if isinstance(corridor, dict) else ()
    start = corridor.get("start_xy_m", ()) if isinstance(corridor, dict) else ()
    end = corridor.get("end_xy_m", ()) if isinstance(corridor, dict) else ()
    goal = guidance.get("local_goal_xy_m", ())
    try:
        center_xy = (float(center[0]), float(center[1]))
        start_xy = (float(start[0]), float(start[1]))
        end_xy = (float(end[0]), float(end[1]))
        goal_xy = (float(goal[0]), float(goal[1]))
        half_width = float(corridor.get("half_width_m", 0.0))
        detour_margin = float(corridor.get("detour_margin_m", 0.0))
    except (IndexError, TypeError, ValueError):
        return {"available": False}
    if direction is None:
        return {"available": False}
    rel_x = payload_xy[0] - center_xy[0]
    rel_y = payload_xy[1] - center_xy[1]
    along = rel_x * direction[0] + rel_y * direction[1]
    cross = abs(-rel_x * direction[1] + rel_y * direction[0])
    local_goal_distance = hypot(payload_xy[0] - goal_xy[0], payload_xy[1] - goal_xy[1])
    start_projection = start_xy[0] * direction[0] + start_xy[1] * direction[1]
    end_projection = end_xy[0] * direction[0] + end_xy[1] * direction[1]
    payload_projection = payload_xy[0] * direction[0] + payload_xy[1] * direction[1]
    along_min = min(start_projection, end_projection) - detour_margin
    along_max = max(start_projection, end_projection) + detour_margin
    inside_lateral = cross <= half_width + detour_margin + 1.0e-9
    inside_longitudinal = along_min <= payload_projection <= along_max
    lateral_violation = max(0.0, cross - half_width - detour_margin)
    longitudinal_violation = max(
        0.0,
        along_min - payload_projection,
        payload_projection - along_max,
    )
    envelope_violation = hypot(lateral_violation, longitudinal_violation)
    return {
        "available": True,
        "segment_index": int(guidance["segment_index"]),
        "segment_id": str(guidance["segment_id"]),
        "payload_xy_m": [round(payload_xy[0], 6), round(payload_xy[1], 6)],
        "along_segment_progress_m": round(along, 6),
        "cross_track_error_m": round(cross, 6),
        "corridor_half_width_m": half_width,
        "detour_margin_m": detour_margin,
        "inside_lateral_envelope": inside_lateral,
        "inside_longitudinal_envelope": inside_longitudinal,
        "inside_corridor_envelope": inside_lateral and inside_longitudinal,
        "lateral_envelope_violation_m": round(lateral_violation, 6),
        "longitudinal_envelope_violation_m": round(longitudinal_violation, 6),
        "corridor_envelope_violation_m": round(envelope_violation, 6),
        "segment_projection_bounds_m": [round(along_min, 6), round(along_max, 6)],
        "payload_segment_projection_m": round(payload_projection, 6),
        "local_goal_distance_m": round(local_goal_distance, 6),
    }


def _completion_gate(task: BoxCarryTask, completed_prefix: int) -> dict[str, Any]:
    """Return the predicate that completes the active transport segment."""

    index = max(0, int(completed_prefix))
    if index < len(task.route_progress_gates):
        return {
            "kind": "oriented_progress_gate",
            **asdict(task.route_progress_gates[index]),
        }
    return {
        "kind": "terminal_target_region",
        "phase_name": "target_region",
        "center_xy_m": [task.target_center_x_m, task.target_center_y_m],
        "xy_tolerance_m": task.target_xy_tolerance_m,
    }


def _payload_xy(runtime_state: dict[str, Any] | None) -> tuple[float, float] | None:
    state = runtime_state if isinstance(runtime_state, dict) else {}
    payload = state.get("payload", {})
    if not isinstance(payload, dict):
        return None
    try:
        x, y = float(payload["x"]), float(payload["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y) if isfinite(x) and isfinite(y) else None


def _fresh_route_observation_after_turn_budget(
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    state = runtime_state if isinstance(runtime_state, dict) else {}
    episode = state.get("turn_episode", {})
    episode = episode if isinstance(episode, dict) else {}
    required = bool(
        str(episode.get("status", "")) == "local_direction_replan_required"
        and episode.get("replan_requires_fresh_global_observation", False)
    )
    if not required:
        return {"required": False, "valid": True}
    observation = state.get("global_observation", {})
    observation = observation if isinstance(observation, dict) else {}
    try:
        captured_at = float(observation.get("captured_at_unix", 0.0))
        previous_capture = float(
            episode.get("route_observation_captured_at_unix", 0.0)
        )
    except (TypeError, ValueError):
        captured_at = 0.0
        previous_capture = 0.0
    valid = bool(
        observation.get("available", False)
        and captured_at > 0.0
        and (previous_capture <= 0.0 or captured_at > previous_capture + 1.0e-6)
    )
    return {
        "required": True,
        "valid": valid,
        "captured_at_unix": captured_at,
        "previous_route_observation_captured_at_unix": previous_capture,
        "source": str(observation.get("source", "")),
        "reason": (
            "fresh global observation is newer than the exhausted turn reference"
            if valid
            else "turn budget was exhausted; capture a newer global observation before validating another route reference"
        ),
    }


def _task_plan_navigation(task_plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = task_plan if isinstance(task_plan, dict) else {}
    subtask = plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    navigation = subtask.get("navigation", {})
    return navigation if isinstance(navigation, dict) else {}


def _segment_gate_intersection(
    segment: RouteSegment,
    gate: RouteProgressGate,
) -> tuple[float, float] | None:
    delta_x = segment.goal_x_m - segment.start_x_m
    delta_y = segment.goal_y_m - segment.start_y_m
    denominator = gate.normal_x * delta_x + gate.normal_y * delta_y
    if abs(denominator) <= ROUTE_GEOMETRY_TOLERANCE_M:
        return None
    numerator = gate.threshold_m - (
        gate.normal_x * segment.start_x_m
        + gate.normal_y * segment.start_y_m
    )
    fraction = numerator / denominator
    if not (
        -ROUTE_GEOMETRY_TOLERANCE_M
        <= fraction
        <= 1.0 + ROUTE_GEOMETRY_TOLERANCE_M
    ):
        return None
    return (
        segment.start_x_m + fraction * delta_x,
        segment.start_y_m + fraction * delta_y,
    )


def _point_cross_track_error(
    segment: RouteSegment,
    point: tuple[float, float],
) -> float:
    rel_x = point[0] - segment.center_x_m
    rel_y = point[1] - segment.center_y_m
    return abs(-rel_x * segment.direction_y + rel_y * segment.direction_x)


def _point_in_segment_envelope(
    segment: RouteSegment,
    point: tuple[float, float],
) -> bool:
    cross = _point_cross_track_error(segment, point)
    projection = point[0] * segment.direction_x + point[1] * segment.direction_y
    start_projection = (
        segment.start_x_m * segment.direction_x
        + segment.start_y_m * segment.direction_y
    )
    goal_projection = (
        segment.goal_x_m * segment.direction_x
        + segment.goal_y_m * segment.direction_y
    )
    return bool(
        cross
        <= segment.half_width_m
        + segment.detour_margin_m
        + ROUTE_GEOMETRY_TOLERANCE_M
        and min(start_projection, goal_projection)
        - segment.detour_margin_m
        - ROUTE_GEOMETRY_TOLERANCE_M
        <= projection
        <= max(start_projection, goal_projection)
        + segment.detour_margin_m
        + ROUTE_GEOMETRY_TOLERANCE_M
    )


def _unit(x: float, y: float) -> tuple[float, float] | None:
    magnitude = hypot(float(x), float(y))
    if not isfinite(magnitude) or magnitude <= 1.0e-9:
        return None
    return float(x) / magnitude, float(y) / magnitude


def _unit_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return _unit(float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


__all__ = [
    "ROUTE_GEOMETRY_TOLERANCE_M",
    "ROUTE_REFERENCE_MIN_DOT",
    "ROUTE_SPEC_SCHEMA_VERSION",
    "RouteSpecConfigurationError",
    "active_route_segment",
    "load_task_route_spec",
    "route_guidance",
    "route_segment_state",
    "route_spec_document",
    "validate_declared_route_reference",
    "validate_locked_route_reference_unchanged",
    "validate_task_route_spec",
    "validate_task_plan_route_reference",
]
