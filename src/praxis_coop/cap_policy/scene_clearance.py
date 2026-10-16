"""Physics-frame clearance checks against the authored wall geometry."""

from __future__ import annotations

from functools import lru_cache
from math import atan2, ceil, cos, degrees, hypot, isfinite, pi, radians, sin
from pathlib import Path
from typing import Any

from praxis_coop.sonic_mujoco.manual_layout_scene import DEFAULT_USDA_SCENE
from praxis_coop.utils.usda_layout import parse_manual_layout_prims


DEFAULT_CLEARANCE_HORIZON_M = 3.0
DEFAULT_WALL_INFLATION_M = 0.20
DEFAULT_PAYLOAD_VERTICAL_SAFETY_HALF_EXTENT_M = 0.10
DEFAULT_PIVOT_SWEEP_ANGLE_DEG = 30.0
DEFAULT_PIVOT_SWEEP_SAMPLE_DEG = 1.0
DEFAULT_ROBOT_FOOTPRINT_HALF_EXTENTS_M = (0.28, 0.22)
DEFAULT_PIVOT_FOOTPRINT_CLEARANCE_MARGIN_M = 0.05
MIN_EXECUTABLE_PIVOT_ANGLE_DEG = 2.0
BLOCKED_CLEARANCE_M = 0.75
OPEN_CLEARANCE_M = 1.25


def authored_scene_motion_clearance(
    runtime_state: dict[str, Any],
    *,
    source_usda: Path = DEFAULT_USDA_SCENE,
    horizon_m: float = DEFAULT_CLEARANCE_HORIZON_M,
    wall_inflation_m: float = DEFAULT_WALL_INFLATION_M,
) -> dict[str, Any]:
    """Project body translations and signed turn headings against scene walls.

    This is a local collision check, not a route or waypoint planner. It uses
    the current MuJoCo robot/payload poses and the tracked authored wall boxes
    so a visual model cannot label an arrow through a wall as open space.
    """

    heading = _mean_robot_heading(runtime_state)
    origins = _team_origins(runtime_state)
    payload_bottom_z = _payload_bottom_z(runtime_state)
    walls = _authored_walls(Path(source_usda))
    if heading is None or not origins or not walls:
        return {
            "available": False,
            "source": "authored_wall_geometry_and_current_mujoco_pose",
            "reason": "current heading, team origins, or authored walls are unavailable",
        }

    horizon = _positive_or_default(horizon_m, DEFAULT_CLEARANCE_HORIZON_M)
    inflation = max(0.0, float(wall_inflation_m))
    translation_yaws = {
        "forward": heading,
        "backward": heading + pi,
        "left": heading + 0.5 * pi,
        "right": heading - 0.5 * pi,
    }
    translations = {
        direction: _candidate_clearance(
            yaw,
            origins=origins,
            walls=walls,
            horizon_m=horizon,
            inflation_m=inflation,
            payload_bottom_z=payload_bottom_z,
        )
        for direction, yaw in translation_yaws.items()
    }
    turns = {
        direction: _pivot_sweep_clearance(
            direction,
            heading=heading,
            origins=origins,
            walls=walls,
            inflation_m=inflation,
            payload_bottom_z=payload_bottom_z,
            requested_angle_deg=DEFAULT_PIVOT_SWEEP_ANGLE_DEG,
        )
        for direction in ("clockwise", "counterclockwise")
    }
    heading_corrections = {
        direction: correction
        for direction, evidence in translations.items()
        if evidence["relation"] == "blocked"
        and (
            correction := _small_heading_correction_for_translation(
                direction,
                heading=heading,
                origins=origins,
                walls=walls,
                horizon_m=horizon,
                inflation_m=inflation,
                payload_bottom_z=payload_bottom_z,
            )
        )
    }
    open_translations = [
        direction
        for direction, evidence in translations.items()
        if evidence["relation"] == "open"
    ]
    payload = runtime_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    current_wall_contacts = tuple(
        str(item)
        for item in (
            payload.get("wall_contacts")
            or contacts.get("walls")
            or ()
        )
    )
    return {
        "available": True,
        "source": "authored_wall_geometry_and_current_mujoco_pose",
        "scope": (
            "predictive local motion-envelope clearance only; this does not "
            "prescribe a route, waypoint, skill, or target direction, and a "
            "blocked prediction does not assert current wall contact"
        ),
        "current_wall_contact": bool(current_wall_contacts),
        "current_wall_contacts": list(current_wall_contacts),
        "prediction_semantics": (
            "translation relation is based on forward swept clearance from the "
            "current trusted pose. Current collision is reported separately from "
            "simulator contacts; executable code is checked against its literal "
            "distance_m before dispatch"
        ),
        "wall_inflation_m": round(inflation, 3),
        "clearance_horizon_m": round(horizon, 3),
        "team_origin_labels": sorted(origins),
        "translation_candidates": translations,
        "turn_candidates": turns,
        "turn_clearance_contract": {
            "trajectory_model": (
                "finite coordinated_pivot sweep: the geometry-derived inner robot "
                "stays translation-stationary while the outer robot and payload rotate "
                "about it"
            ),
            "requested_angle_deg": DEFAULT_PIVOT_SWEEP_ANGLE_DEG,
            "minimum_executable_angle_deg": MIN_EXECUTABLE_PIVOT_ANGLE_DEG,
            "rule": (
                "A signed turn is blocked only when even the minimum executable "
                "same-sign pivot angle intersects inflated authored-wall geometry. "
                "When max_safe_angle_deg is smaller than the requested angle, retain "
                "the signed direction and cap the executable angle instead of "
                "switching turn direction."
            ),
        },
        "small_heading_corrections_for_blocked_translation": heading_corrections,
        "open_translation_directions": open_translations,
        "authority": (
            "Exact authored-wall geometry overrides contradictory visual labels "
            "for predicted motion clearance. Simulator contacts remain authoritative "
            "for whether a wall collision exists now."
        ),
    }


def candidate_is_blocked(
    clearance: object,
    *,
    family: str,
    objective: str,
) -> bool:
    """Return whether one host-projected room-scale candidate has wall risk.

    This is a predictive envelope label, never a claim that the current trusted
    checkpoint is already in contact with a wall.
    """

    report = clearance if isinstance(clearance, dict) else {}
    candidates = report.get(
        "turn_candidates" if family == "turn" else "translation_candidates",
        {},
    )
    candidates = candidates if isinstance(candidates, dict) else {}
    evidence = candidates.get(objective, {})
    return bool(
        isinstance(evidence, dict)
        and evidence.get("relation") == "blocked"
    )


def maximum_safe_turn_angle_deg(
    clearance: object,
    turn_direction: str,
) -> float | None:
    """Return the contiguous wall-clear pivot angle for one signed direction."""

    report = clearance if isinstance(clearance, dict) else {}
    candidates = report.get("turn_candidates", {})
    candidates = candidates if isinstance(candidates, dict) else {}
    evidence = candidates.get(str(turn_direction).strip().lower(), {})
    if not isinstance(evidence, dict):
        return None
    try:
        value = float(evidence["max_safe_angle_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if isfinite(value) and value >= 0.0 else None


def open_translation_directions(clearance: object) -> tuple[str, ...]:
    """Return body-frame directions with at least the open-clearance threshold."""

    report = clearance if isinstance(clearance, dict) else {}
    values = report.get("open_translation_directions", ())
    if not isinstance(values, (list, tuple)):
        return ()
    return tuple(
        str(value)
        for value in values
        if str(value) in {"forward", "backward", "left", "right"}
    )


def unique_small_heading_correction(
    clearance: object,
) -> dict[str, Any]:
    """Return one unambiguous local correction, otherwise no recommendation."""

    report = clearance if isinstance(clearance, dict) else {}
    corrections = report.get(
        "small_heading_corrections_for_blocked_translation",
        {},
    )
    corrections = corrections if isinstance(corrections, dict) else {}
    if len(corrections) != 1:
        return {}
    desired_direction, correction = next(iter(corrections.items()))
    if not isinstance(correction, dict):
        return {}
    return {
        "desired_translation_direction": str(desired_direction),
        **correction,
    }


def pivot_role_assignment(
    runtime_state: dict[str, Any],
    turn_direction: str,
) -> dict[str, str]:
    """Map one signed turn to immutable front/back robot IDs."""

    direction = str(turn_direction).strip().lower()
    if direction not in {"clockwise", "counterclockwise"}:
        return {}
    heading = _mean_robot_heading(runtime_state)
    origins = _team_origins(runtime_state)
    if heading is None:
        return {}
    inner_robot, outer_robot = _turn_roles(
        direction,
        heading=heading,
        origins=origins,
    )
    if not inner_robot or not outer_robot:
        return {}
    return {
        inner_robot: "inner",
        outer_robot: "outer",
    }


def pivot_candidate_swept_clearance(
    runtime_state: dict[str, Any],
    *,
    turn_direction: str,
    angle_deg: float,
    roles: dict[str, str] | None = None,
    source_usda: Path = DEFAULT_USDA_SCENE,
    wall_clearance_margin_m: float = DEFAULT_PIVOT_FOOTPRINT_CLEARANCE_MARGIN_M,
    hard_collision_margin_m: float | None = None,
) -> dict[str, Any]:
    """Check one model-authored pivot over the complete team footprint.

    Direction, magnitude, and temporary roles are inputs. This verifier never
    selects a route or replacement maneuver; it only evaluates the exact
    candidate at the current trusted pose.
    """

    direction = str(turn_direction).strip().lower()
    heading = _mean_robot_heading(runtime_state)
    origins = _team_origins(runtime_state)
    walls = _authored_walls(Path(source_usda))
    if direction not in {"clockwise", "counterclockwise"}:
        return {
            "available": False,
            "relation": "unavailable",
            "executable": False,
            "reason": "turn direction must be clockwise or counterclockwise",
        }
    if heading is None or not origins or not walls:
        return {
            "available": False,
            "relation": "unavailable",
            "executable": False,
            "reason": "current heading, team poses, or authored walls are unavailable",
        }

    role_mapping = dict(roles or pivot_role_assignment(runtime_state, direction))
    inner = [robot for robot, role in role_mapping.items() if role == "inner"]
    outer = [robot for robot, role in role_mapping.items() if role == "outer"]
    if len(inner) != 1 or len(outer) != 1 or set(inner + outer) != {"front", "back"}:
        return {
            "available": False,
            "relation": "unavailable",
            "executable": False,
            "reason": "candidate requires exactly one front/back inner and outer role",
            "candidate_roles": role_mapping,
        }

    footprints = _team_footprints(runtime_state, origins=origins)
    missing = sorted({"payload", "front", "back"} - set(footprints))
    if missing:
        return {
            "available": False,
            "relation": "unavailable",
            "executable": False,
            "reason": (
                "complete payload/front/back footprint telemetry is required for "
                "candidate-specific pivot clearance"
            ),
            "missing_footprints": missing,
            "footprint_labels": sorted(footprints),
        }

    quality_margin = max(0.0, float(wall_clearance_margin_m))
    hard_margin = (
        quality_margin
        if hard_collision_margin_m is None
        else max(0.0, float(hard_collision_margin_m))
    )
    result = _full_team_pivot_sweep_clearance(
        direction,
        heading=heading,
        origins=origins,
        walls=walls,
        clearance_margin_m=hard_margin,
        payload_bottom_z=_payload_bottom_z(runtime_state),
        requested_angle_deg=float(angle_deg),
        footprints=footprints,
        inner_robot=inner[0],
    )
    clearance_m = float(result.get("clearance_m", 0.0))
    hard_collision = result.get("relation") != "open"
    return {
        "available": True,
        "candidate_turn_direction": direction,
        "candidate_angle_deg": float(angle_deg),
        "candidate_roles": role_mapping,
        "footprint_labels": sorted(footprints),
        "complete_team_footprint": True,
        "executable": not hard_collision,
        "hard_collision": hard_collision,
        "hard_collision_margin_m": hard_margin,
        "quality_clearance_margin_m": quality_margin,
        "low_clearance_warning": bool(
            not hard_collision and clearance_m < quality_margin
        ),
        **result,
    }


def _small_heading_correction_for_translation(
    translation_direction: str,
    *,
    heading: float,
    origins: dict[str, tuple[float, float]],
    walls: tuple[tuple[str, float, float, float, float, float, float, float], ...],
    horizon_m: float,
    inflation_m: float,
    payload_bottom_z: float | None,
) -> dict[str, Any]:
    axis_offsets = {
        "forward": 0.0,
        "backward": pi,
        "left": 0.5 * pi,
        "right": -0.5 * pi,
    }
    if translation_direction not in axis_offsets:
        return {}
    for angle_deg in (5.0, 10.0, 15.0):
        candidates: list[tuple[str, dict[str, Any]]] = []
        for turn_direction, sign in (
            ("clockwise", -1.0),
            ("counterclockwise", 1.0),
        ):
            corrected_heading = heading + sign * radians(angle_deg)
            evidence = _candidate_clearance(
                corrected_heading + axis_offsets[translation_direction],
                origins=origins,
                walls=walls,
                horizon_m=horizon_m,
                inflation_m=inflation_m,
                payload_bottom_z=payload_bottom_z,
            )
            if evidence["relation"] == "open":
                candidates.append((turn_direction, evidence))
        if not candidates:
            continue
        turn_direction, evidence = max(
            candidates,
            key=lambda item: float(item[1]["clearance_m"]),
        )
        inner_robot, outer_robot = _turn_roles(
            turn_direction,
            heading=heading,
            origins=origins,
        )
        if not inner_robot or not outer_robot:
            return {}
        return {
            "turn_direction": turn_direction,
            "angle_deg": angle_deg,
            "resulting_translation_clearance_m": evidence["clearance_m"],
            "resulting_translation_relation": evidence["relation"],
            "inner_robot": inner_robot,
            "outer_robot": outer_robot,
            "basis": (
                "smallest sampled shared-yaw correction that makes the blocked "
                "body-frame translation locally wall-clear"
            ),
        }
    return {}


def _turn_roles(
    turn_direction: str,
    *,
    heading: float,
    origins: dict[str, tuple[float, float]],
) -> tuple[str, str]:
    if "front" not in origins or "back" not in origins:
        return "", ""
    center_x = 0.5 * (origins["front"][0] + origins["back"][0])
    center_y = 0.5 * (origins["front"][1] + origins["back"][1])
    body_left = (-sin(heading), cos(heading))
    lateral = {
        robot_id: (
            (origins[robot_id][0] - center_x) * body_left[0]
            + (origins[robot_id][1] - center_y) * body_left[1]
        )
        for robot_id in ("front", "back")
    }
    if turn_direction == "counterclockwise":
        inner = max(lateral, key=lateral.get)
    else:
        inner = min(lateral, key=lateral.get)
    outer = "back" if inner == "front" else "front"
    return inner, outer


def _candidate_clearance(
    yaw_rad: float,
    *,
    origins: dict[str, tuple[float, float]],
    walls: tuple[tuple[str, float, float, float, float, float, float, float], ...],
    horizon_m: float,
    inflation_m: float,
    payload_bottom_z: float | None,
) -> dict[str, Any]:
    direction = (cos(yaw_rad), sin(yaw_rad))
    nearest_distance = horizon_m
    nearest_wall = ""
    limiting_origin = ""
    for origin_label, origin_xy in origins.items():
        for wall_name, x, y, z, sx, sy, sz, yaw in walls:
            if (
                origin_label == "payload"
                and payload_bottom_z is not None
                and payload_bottom_z > z + 0.5 * sz
            ):
                continue
            distance = _ray_oriented_box_distance(
                origin_xy,
                direction,
                center_xy=(x, y),
                size_xy=(sx, sy),
                yaw_rad=yaw,
                inflation_m=inflation_m,
            )
            if distance is None or distance >= nearest_distance:
                continue
            nearest_distance = distance
            nearest_wall = wall_name
            limiting_origin = origin_label
    return {
        "world_yaw_deg": round(_wrap_degrees(degrees(yaw_rad)), 3),
        "world_direction_xy": [
            round(direction[0], 4),
            round(direction[1], 4),
        ],
        "clearance_m": round(nearest_distance, 3),
        "relation": _clearance_relation(nearest_distance),
        "nearest_wall": nearest_wall,
        "limiting_origin": limiting_origin,
    }


def _pivot_sweep_clearance(
    turn_direction: str,
    *,
    heading: float,
    origins: dict[str, tuple[float, float]],
    walls: tuple[tuple[str, float, float, float, float, float, float, float], ...],
    inflation_m: float,
    payload_bottom_z: float | None,
    requested_angle_deg: float,
    sample_angle_deg: float = DEFAULT_PIVOT_SWEEP_SAMPLE_DEG,
) -> dict[str, Any]:
    """Check the finite trajectory compiled by ``coordinated_pivot``.

    The inner robot is a translation-stationary pivot. The outer robot and the
    supported payload sweep around that live pivot using their current radii.
    This deliberately differs from translation clearance: projecting all three
    origins along the post-turn heading would invent motion for the inner robot
    and can reject the only physically correct cornering direction.
    """

    direction = str(turn_direction).strip().lower()
    sign = -1.0 if direction == "clockwise" else 1.0
    requested = max(0.0, float(requested_angle_deg))
    sample = max(0.1, float(sample_angle_deg))
    inner_robot, outer_robot = _turn_roles(
        direction,
        heading=heading,
        origins=origins,
    )
    if not inner_robot or not outer_robot:
        return {
            "relation": "blocked",
            "requested_angle_deg": round(requested, 3),
            "max_safe_angle_deg": 0.0,
            "minimum_executable_angle_deg": MIN_EXECUTABLE_PIVOT_ANGLE_DEG,
            "reason": "front/back pivot roles are unavailable",
        }

    pivot_xy = origins[inner_robot]
    moving_origins = {
        label: point
        for label, point in origins.items()
        if label != inner_robot
    }
    sample_count = max(1, ceil(requested / sample))
    sample_angles = [
        requested * index / sample_count for index in range(sample_count + 1)
    ]
    nearest_clearance = float("inf")
    nearest_wall = ""
    limiting_origin = ""
    max_safe_angle = 0.0
    first_blocked_angle: float | None = None

    # Match translation-ray behavior for a trusted state already inside only a
    # conservative inflation band: do not call its stationary start a collision.
    wall_inflation_by_origin: dict[tuple[str, str], float] = {}
    for origin_label, origin_xy in origins.items():
        for wall_name, x, y, _z, sx, sy, _sz, yaw in walls:
            inside_inflated = _point_inside_oriented_box(
                origin_xy,
                center_xy=(x, y),
                size_xy=(sx, sy),
                yaw_rad=yaw,
                inflation_m=inflation_m,
            )
            inside_physical = _point_inside_oriented_box(
                origin_xy,
                center_xy=(x, y),
                size_xy=(sx, sy),
                yaw_rad=yaw,
                inflation_m=0.0,
            )
            wall_inflation_by_origin[(origin_label, wall_name)] = (
                0.0 if inside_inflated and not inside_physical else inflation_m
            )

    for angle_deg in sample_angles:
        angle_rad = sign * radians(angle_deg)
        c = cos(angle_rad)
        s = sin(angle_rad)
        sampled_origins = {inner_robot: pivot_xy}
        for origin_label, origin_xy in moving_origins.items():
            rel_x = origin_xy[0] - pivot_xy[0]
            rel_y = origin_xy[1] - pivot_xy[1]
            sampled_origins[origin_label] = (
                pivot_xy[0] + c * rel_x - s * rel_y,
                pivot_xy[1] + s * rel_x + c * rel_y,
            )

        sample_blocked = False
        for origin_label, origin_xy in sampled_origins.items():
            for wall_name, x, y, z, sx, sy, sz, yaw in walls:
                if (
                    origin_label == "payload"
                    and payload_bottom_z is not None
                    and payload_bottom_z > z + 0.5 * sz
                ):
                    continue
                effective_inflation = wall_inflation_by_origin[
                    (origin_label, wall_name)
                ]
                clearance_m = _point_oriented_box_clearance(
                    origin_xy,
                    center_xy=(x, y),
                    size_xy=(sx, sy),
                    yaw_rad=yaw,
                    inflation_m=effective_inflation,
                )
                if clearance_m < nearest_clearance:
                    nearest_clearance = clearance_m
                    nearest_wall = wall_name
                    limiting_origin = origin_label
                if clearance_m <= 1.0e-9:
                    sample_blocked = True
        if sample_blocked:
            first_blocked_angle = angle_deg
            break
        max_safe_angle = angle_deg

    if not isfinite(nearest_clearance):
        nearest_clearance = 0.0
    if max_safe_angle + 1.0e-9 < MIN_EXECUTABLE_PIVOT_ANGLE_DEG:
        relation = "blocked"
    elif max_safe_angle + 1.0e-9 < requested:
        relation = "constrained"
    else:
        relation = "open"
    final_heading = heading + sign * radians(min(requested, max_safe_angle))
    return {
        "world_yaw_deg": round(_wrap_degrees(degrees(final_heading)), 3),
        "requested_angle_deg": round(requested, 3),
        "max_safe_angle_deg": round(max_safe_angle, 3),
        "minimum_executable_angle_deg": MIN_EXECUTABLE_PIVOT_ANGLE_DEG,
        "first_blocked_sample_angle_deg": (
            None if first_blocked_angle is None else round(first_blocked_angle, 3)
        ),
        "clearance_m": round(nearest_clearance, 3),
        "relation": relation,
        "nearest_wall": nearest_wall,
        "limiting_origin": limiting_origin,
        "inner_robot": inner_robot,
        "outer_robot": outer_robot,
        "pivot_world_xy": [round(pivot_xy[0], 4), round(pivot_xy[1], 4)],
        "trajectory_model": "finite_inner_stationary_outer_payload_pivot_sweep",
    }


def _full_team_pivot_sweep_clearance(
    turn_direction: str,
    *,
    heading: float,
    origins: dict[str, tuple[float, float]],
    walls: tuple[tuple[str, float, float, float, float, float, float, float], ...],
    clearance_margin_m: float,
    payload_bottom_z: float | None,
    requested_angle_deg: float,
    footprints: dict[str, dict[str, Any]],
    inner_robot: str,
    sample_angle_deg: float = DEFAULT_PIVOT_SWEEP_SAMPLE_DEG,
) -> dict[str, Any]:
    """Check the payload and both robot OBBs along one exact pivot."""

    direction = str(turn_direction).strip().lower()
    sign = -1.0 if direction == "clockwise" else 1.0
    requested = max(0.0, float(requested_angle_deg))
    sample = max(0.1, float(sample_angle_deg))
    if inner_robot not in {"front", "back"}:
        return {
            "relation": "blocked",
            "requested_angle_deg": round(requested, 3),
            "max_safe_angle_deg": 0.0,
            "minimum_executable_angle_deg": MIN_EXECUTABLE_PIVOT_ANGLE_DEG,
            "reason": "front/back pivot roles are unavailable",
        }
    outer_robot = "back" if inner_robot == "front" else "front"
    if not {inner_robot, outer_robot, "payload"}.issubset(origins):
        return {
            "relation": "blocked",
            "requested_angle_deg": round(requested, 3),
            "max_safe_angle_deg": 0.0,
            "minimum_executable_angle_deg": MIN_EXECUTABLE_PIVOT_ANGLE_DEG,
            "reason": "front/back/payload origins are unavailable",
        }

    pivot_xy = origins[inner_robot]
    sample_count = max(1, ceil(requested / sample))
    sample_angles = [
        requested * index / sample_count for index in range(sample_count + 1)
    ]
    nearest_clearance = float("inf")
    nearest_wall = ""
    limiting_component = ""
    max_safe_angle = 0.0
    first_blocked_angle: float | None = None

    for angle_deg in sample_angles:
        angle_rad = sign * radians(angle_deg)
        c = cos(angle_rad)
        s = sin(angle_rad)
        sampled_origins = {inner_robot: pivot_xy}
        for label, origin_xy in origins.items():
            if label == inner_robot:
                continue
            rel_x = origin_xy[0] - pivot_xy[0]
            rel_y = origin_xy[1] - pivot_xy[1]
            sampled_origins[label] = (
                pivot_xy[0] + c * rel_x - s * rel_y,
                pivot_xy[1] + s * rel_x + c * rel_y,
            )

        sample_blocked = False
        for label, origin_xy in sampled_origins.items():
            footprint = footprints.get(label, {})
            half_extents = footprint.get("half_extents_xy_m", (0.0, 0.0))
            try:
                footprint_half_extents = (
                    max(0.0, float(half_extents[0])),
                    max(0.0, float(half_extents[1])),
                )
                footprint_yaw = float(footprint.get("yaw_rad", heading)) + angle_rad
            except (IndexError, TypeError, ValueError):
                footprint_half_extents = (0.0, 0.0)
                footprint_yaw = heading + angle_rad
            for wall_name, x, y, z, sx, sy, sz, yaw in walls:
                if (
                    label == "payload"
                    and payload_bottom_z is not None
                    and payload_bottom_z > z + 0.5 * sz
                ):
                    continue
                clearance_m = _footprint_oriented_box_clearance(
                    footprint_center_xy=origin_xy,
                    footprint_half_extents_xy=footprint_half_extents,
                    footprint_yaw_rad=footprint_yaw,
                    wall_center_xy=(x, y),
                    wall_size_xy=(sx, sy),
                    wall_yaw_rad=yaw,
                )
                if clearance_m < nearest_clearance:
                    nearest_clearance = clearance_m
                    nearest_wall = wall_name
                    limiting_component = label
                if clearance_m <= clearance_margin_m + 1.0e-9:
                    sample_blocked = True
        if sample_blocked:
            first_blocked_angle = angle_deg
            break
        max_safe_angle = angle_deg

    if not isfinite(nearest_clearance):
        nearest_clearance = clearance_margin_m
    if max_safe_angle + 1.0e-9 < MIN_EXECUTABLE_PIVOT_ANGLE_DEG:
        relation = "blocked"
    elif max_safe_angle + 1.0e-9 < requested:
        relation = "constrained"
    else:
        relation = "open"
    final_heading = heading + sign * radians(min(requested, max_safe_angle))
    return {
        "world_yaw_deg": round(_wrap_degrees(degrees(final_heading)), 3),
        "requested_angle_deg": round(requested, 3),
        "max_safe_angle_deg": round(max_safe_angle, 3),
        "minimum_executable_angle_deg": MIN_EXECUTABLE_PIVOT_ANGLE_DEG,
        "first_blocked_sample_angle_deg": (
            None if first_blocked_angle is None else round(first_blocked_angle, 3)
        ),
        "clearance_m": round(nearest_clearance, 3),
        "required_clearance_margin_m": round(clearance_margin_m, 3),
        "clearance_margin_m": round(nearest_clearance - clearance_margin_m, 3),
        "relation": relation,
        "nearest_wall": nearest_wall,
        "limiting_origin": limiting_component,
        "limiting_component": limiting_component,
        "inner_robot": inner_robot,
        "outer_robot": outer_robot,
        "pivot_world_xy": [round(pivot_xy[0], 4), round(pivot_xy[1], 4)],
        "trajectory_model": "finite_full_team_footprint_pivot_sweep",
    }


def _point_inside_oriented_box(
    point_xy: tuple[float, float],
    *,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_rad: float,
    inflation_m: float,
) -> bool:
    return (
        _point_oriented_box_clearance(
            point_xy,
            center_xy=center_xy,
            size_xy=size_xy,
            yaw_rad=yaw_rad,
            inflation_m=inflation_m,
        )
        <= 1.0e-9
    )


def _point_oriented_box_clearance(
    point_xy: tuple[float, float],
    *,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_rad: float,
    inflation_m: float,
) -> float:
    """Return unsigned point clearance to an inflated oriented rectangle."""

    c = cos(yaw_rad)
    s = sin(yaw_rad)
    rel_x = point_xy[0] - center_xy[0]
    rel_y = point_xy[1] - center_xy[1]
    local_x = c * rel_x + s * rel_y
    local_y = -s * rel_x + c * rel_y
    half_x = 0.5 * abs(size_xy[0]) + max(0.0, float(inflation_m))
    half_y = 0.5 * abs(size_xy[1]) + max(0.0, float(inflation_m))
    outside_x = max(0.0, abs(local_x) - half_x)
    outside_y = max(0.0, abs(local_y) - half_y)
    return hypot(outside_x, outside_y)


def _team_footprints(
    runtime_state: dict[str, Any],
    *,
    origins: dict[str, tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    """Return live oriented ground footprints for available team components."""

    footprints: dict[str, dict[str, Any]] = {}
    payload = runtime_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    raw_extents = payload.get("half_extents_xyz_m", ())
    if "payload" in origins and isinstance(raw_extents, (list, tuple)):
        try:
            payload_half_extents = (
                max(0.0, float(raw_extents[0])),
                max(0.0, float(raw_extents[1])),
            )
        except (IndexError, TypeError, ValueError):
            payload_half_extents = (0.0, 0.0)
        if max(payload_half_extents) > 0.0:
            footprints["payload"] = {
                "half_extents_xy_m": payload_half_extents,
                "yaw_rad": _mapping_yaw(payload, default=0.0),
                "source": "runtime_payload_obb",
            }

    robots = runtime_state.get("robots", {})
    robots = robots if isinstance(robots, dict) else {}
    for robot_id in ("front", "back"):
        if robot_id not in origins:
            continue
        robot = robots.get(robot_id, {})
        robot = robot if isinstance(robot, dict) else {}
        pose = robot.get("root_pose", robot)
        pose = pose if isinstance(pose, dict) else robot
        footprints[robot_id] = {
            "half_extents_xy_m": DEFAULT_ROBOT_FOOTPRINT_HALF_EXTENTS_M,
            "yaw_rad": _mapping_yaw(pose, default=0.0),
            "source": "generic_robot_pelvis_and_feet_envelope",
        }
    return footprints


def _mapping_yaw(mapping: dict[str, Any], *, default: float) -> float:
    raw = mapping.get("yaw_rad")
    if raw is None and mapping.get("yaw_deg") is not None:
        try:
            raw = radians(float(mapping["yaw_deg"]))
        except (TypeError, ValueError):
            raw = None
    try:
        yaw = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return yaw if isfinite(yaw) else float(default)


def _footprint_oriented_box_clearance(
    *,
    footprint_center_xy: tuple[float, float],
    footprint_half_extents_xy: tuple[float, float],
    footprint_yaw_rad: float,
    wall_center_xy: tuple[float, float],
    wall_size_xy: tuple[float, float],
    wall_yaw_rad: float,
) -> float:
    """Return Euclidean clearance between one footprint OBB and wall OBB."""

    if max(footprint_half_extents_xy) <= 1.0e-9:
        return _point_oriented_box_clearance(
            footprint_center_xy,
            center_xy=wall_center_xy,
            size_xy=wall_size_xy,
            yaw_rad=wall_yaw_rad,
            inflation_m=0.0,
        )
    footprint = _oriented_rectangle_vertices(
        footprint_center_xy,
        footprint_half_extents_xy,
        footprint_yaw_rad,
    )
    wall = _oriented_rectangle_vertices(
        wall_center_xy,
        (0.5 * abs(wall_size_xy[0]), 0.5 * abs(wall_size_xy[1])),
        wall_yaw_rad,
    )
    if _convex_polygons_intersect(footprint, wall):
        return 0.0
    return min(
        _point_segment_distance(point, start, end)
        for polygon, other in ((footprint, wall), (wall, footprint))
        for point in polygon
        for start, end in zip(other, (*other[1:], other[0]), strict=True)
    )


def _oriented_rectangle_vertices(
    center_xy: tuple[float, float],
    half_extents_xy: tuple[float, float],
    yaw_rad: float,
) -> tuple[tuple[float, float], ...]:
    c = cos(yaw_rad)
    s = sin(yaw_rad)
    half_x, half_y = half_extents_xy
    return tuple(
        (
            center_xy[0] + c * local_x - s * local_y,
            center_xy[1] + s * local_x + c * local_y,
        )
        for local_x, local_y in (
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y),
        )
    )


def _convex_polygons_intersect(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> bool:
    for polygon in (first, second):
        for start, end in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
            edge_x = end[0] - start[0]
            edge_y = end[1] - start[1]
            axis = (-edge_y, edge_x)
            norm = hypot(*axis)
            if norm <= 1.0e-12:
                continue
            unit = (axis[0] / norm, axis[1] / norm)
            first_projection = [
                point[0] * unit[0] + point[1] * unit[1] for point in first
            ]
            second_projection = [
                point[0] * unit[0] + point[1] * unit[1] for point in second
            ]
            if max(first_projection) < min(second_projection) - 1.0e-9 or max(
                second_projection
            ) < min(first_projection) - 1.0e-9:
                return False
    return True


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    edge_x = end[0] - start[0]
    edge_y = end[1] - start[1]
    denominator = edge_x * edge_x + edge_y * edge_y
    if denominator <= 1.0e-18:
        return hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * edge_x
        + (point[1] - start[1]) * edge_y
    ) / denominator
    projection = min(1.0, max(0.0, projection))
    nearest = (start[0] + projection * edge_x, start[1] + projection * edge_y)
    return hypot(point[0] - nearest[0], point[1] - nearest[1])


def _ray_oriented_box_distance(
    origin_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    *,
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    yaw_rad: float,
    inflation_m: float,
) -> float | None:
    """Return first forward ray intersection with an inflated oriented box."""

    c = cos(yaw_rad)
    s = sin(yaw_rad)
    rel_x = origin_xy[0] - center_xy[0]
    rel_y = origin_xy[1] - center_xy[1]
    local_origin = (
        c * rel_x + s * rel_y,
        -s * rel_x + c * rel_y,
    )
    local_direction = (
        c * direction_xy[0] + s * direction_xy[1],
        -s * direction_xy[0] + c * direction_xy[1],
    )
    physical_half_sizes = (
        0.5 * abs(size_xy[0]),
        0.5 * abs(size_xy[1]),
    )
    inflated_half_sizes = (
        0.5 * abs(size_xy[0]) + inflation_m,
        0.5 * abs(size_xy[1]) + inflation_m,
    )
    inside_inflated = all(
        abs(origin) <= half_size
        for origin, half_size in zip(
            local_origin,
            inflated_half_sizes,
            strict=True,
        )
    )
    inside_physical = all(
        abs(origin) <= half_size
        for origin, half_size in zip(
            local_origin,
            physical_half_sizes,
            strict=True,
        )
    )
    # A robot or payload origin can sit inside the conservative inflation band
    # without touching the authored wall. In that case, test the ray against
    # the real wall: motion toward it remains blocked, while motion away may
    # leave the margin instead of being reported as zero-clearance in every
    # direction.
    half_sizes = (
        physical_half_sizes
        if inside_inflated and not inside_physical
        else inflated_half_sizes
    )
    t_min = float("-inf")
    t_max = float("inf")
    for origin, direction, half_size in zip(
        local_origin,
        local_direction,
        half_sizes,
        strict=True,
    ):
        if abs(direction) <= 1.0e-9:
            if origin < -half_size or origin > half_size:
                return None
            continue
        t1 = (-half_size - origin) / direction
        t2 = (half_size - origin) / direction
        t_near, t_far = sorted((t1, t2))
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)
        if t_min > t_max:
            return None
    if t_max < 0.0:
        return None
    return max(0.0, t_min)


def _clearance_relation(clearance_m: float) -> str:
    if clearance_m <= BLOCKED_CLEARANCE_M:
        return "blocked"
    if clearance_m < OPEN_CLEARANCE_M:
        return "constrained"
    return "open"


def _mean_robot_heading(runtime_state: dict[str, Any]) -> float | None:
    raw_mean = runtime_state.get("mean_heading_yaw_rad")
    try:
        mean = float(raw_mean)
    except (TypeError, ValueError):
        mean = float("nan")
    if isfinite(mean):
        return mean

    robots = runtime_state.get("robots", {})
    if not isinstance(robots, dict):
        return None
    samples: list[float] = []
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            continue
        pose = robot.get("root_pose", robot)
        pose = pose if isinstance(pose, dict) else robot
        raw_yaw = pose.get("yaw_rad")
        if raw_yaw is None and pose.get("yaw_deg") is not None:
            try:
                raw_yaw = radians(float(pose["yaw_deg"]))
            except (TypeError, ValueError):
                raw_yaw = None
        try:
            yaw = float(raw_yaw)
        except (TypeError, ValueError):
            continue
        if isfinite(yaw):
            samples.append(yaw)
    if not samples:
        return None
    return atan2(
        sum(sin(value) for value in samples),
        sum(cos(value) for value in samples),
    )


def _team_origins(
    runtime_state: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    origins: dict[str, tuple[float, float]] = {}
    payload = runtime_state.get("payload", {})
    if isinstance(payload, dict):
        point = _xy_from_mapping(payload)
        if point is not None:
            origins["payload"] = point
    robots = runtime_state.get("robots", {})
    if isinstance(robots, dict):
        for robot_id in ("front", "back"):
            robot = robots.get(robot_id, {})
            if not isinstance(robot, dict):
                continue
            pose = robot.get("root_pose", robot)
            point = _xy_from_mapping(pose if isinstance(pose, dict) else robot)
            if point is not None:
                origins[robot_id] = point
    return origins


def _payload_bottom_z(runtime_state: dict[str, Any]) -> float | None:
    """Estimate the carried box bottom for low-wall clearance filtering."""

    payload = runtime_state.get("payload", {})
    if not isinstance(payload, dict):
        return None
    try:
        center_z = float(payload["z"])
        top_corner_range = max(
            0.0,
            float(payload.get("top_surface_corner_z_range_m", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not isfinite(center_z) or not isfinite(top_corner_range):
        return None
    # The runtime exposes top-face corner range but not the full geom extent.
    # Add a conservative half-thickness allowance so a box that is genuinely
    # above a low wall does not block every 2-D ray through its COM projection.
    return (
        center_z
        - 0.5 * top_corner_range
        - DEFAULT_PAYLOAD_VERTICAL_SAFETY_HALF_EXTENT_M
    )


def _xy_from_mapping(mapping: dict[str, Any]) -> tuple[float, float] | None:
    try:
        x = float(mapping["x"])
        y = float(mapping["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not isfinite(x) or not isfinite(y):
        return None
    return x, y


@lru_cache(maxsize=4)
def _authored_walls(
    source_usda: Path,
) -> tuple[
    tuple[str, float, float, float, float, float, float, float], ...
]:
    return tuple(
        (
            prim.name,
            float(prim.x),
            float(prim.y),
            float(prim.z),
            abs(float(prim.sx)),
            abs(float(prim.sy)),
            abs(float(prim.sz)),
            float(prim.yaw),
        )
        for prim in parse_manual_layout_prims(source_usda)
        if prim.kind == "wall"
    )


def _positive_or_default(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) and parsed > 0.0 else default


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


__all__ = [
    "authored_scene_motion_clearance",
    "candidate_is_blocked",
    "maximum_safe_turn_angle_deg",
    "open_translation_directions",
    "pivot_candidate_swept_clearance",
    "pivot_role_assignment",
    "unique_small_heading_correction",
]
