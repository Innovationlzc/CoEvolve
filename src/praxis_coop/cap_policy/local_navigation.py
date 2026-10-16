"""Scene-independent local heading contracts for cooperative transport.

The Task Orchestrator owns the route and declares what the current subtask is
trying to approach.  This module only converts that declaration and live
MuJoCo poses into local, body-frame navigation evidence.  It deliberately has
no knowledge of corridor names, waypoint indices, or authored map axes.
"""

from __future__ import annotations

from math import atan2, cos, degrees, hypot, isfinite, radians, sin
from typing import Any


LOCAL_NAVIGATION_SCHEMA_VERSION = 1
LOCAL_NAVIGATION_CONTRACT_KEY = "local_navigation"
FINAL_TARGET_REFERENCE = "final_target"
RUNTIME_LOCAL_GOAL_REFERENCE = "runtime_local_goal"
VISIBLE_PASSAGE_REFERENCE = "visible_connected_free_space"
FORMATION_REFERENCE = "formation_only"
PAYLOAD_LIFECYCLE_REFERENCE = "payload_lifecycle"

HEADING_ALIGNMENT_TOLERANCE_DEG = 30.0
MIN_HEADING_IMPROVEMENT_DEG = 1.0
AMBIGUOUS_REAR_HEADING_DEG = 5.0

TURN_METHODS = frozenset(
    {
        "coordinated_arc_transport",
        "coordinated_pivot",
        "carry_turn_in_place",
        "turn_clockwise",
        "turn_counterclockwise",
    }
)
LINEAR_METHODS = frozenset(
    {
        "coordinated_linear_transport",
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
    }
)


def build_local_navigation_contract(
    task_plan: dict[str, Any] | None,
    runtime_state: dict[str, Any] | None,
    *,
    heading_tolerance_deg: float = HEADING_ALIGNMENT_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Build one local contract from a model-owned subtask and live poses.

    An explicit ``current_subtask.navigation.reference`` is preferred.  The
    compatibility inference only recognizes a generic target-approach relation:
    the current phase immediately precedes payload placement/release.  It does
    not recognize benchmark-specific phase IDs.
    """

    plan = task_plan if isinstance(task_plan, dict) else {}
    state = runtime_state if isinstance(runtime_state, dict) else {}
    reference, reference_source = _navigation_reference(plan)
    if reference not in {
        FINAL_TARGET_REFERENCE,
        RUNTIME_LOCAL_GOAL_REFERENCE,
        VISIBLE_PASSAGE_REFERENCE,
    }:
        return {}
    heading_deg = team_heading_deg(state)
    payload_xy = _payload_xy(state)
    if heading_deg is None or payload_xy is None:
        return {}
    payload_x, payload_y = payload_xy

    navigation = _subtask_navigation(plan)
    declared_direction = _world_direction_xy(navigation.get("world_direction_xy"))
    locked_direction = _locked_navigation_direction(state, reference=reference)
    direction_mismatch = bool(
        locked_direction
        and declared_direction
        and _direction_dot(locked_direction, declared_direction) < 0.995
    )
    if reference == VISIBLE_PASSAGE_REFERENCE:
        direction = locked_direction or declared_direction
        if direction is None:
            return {}
        bearing_deg = degrees(atan2(direction[1], direction[0]))
        error_deg = wrap_degrees(bearing_deg - heading_deg)
        distance_m: float | None = None
        goal_source = (
            "runtime_state.turn_episode.locked_world_direction_xy"
            if locked_direction
            else "task_orchestrator.current_subtask.navigation.world_direction_xy"
        )
        reference_geometry = "world_direction"
        goal_world_xy: list[float] = []
        goal_world_direction = [round(direction[0], 6), round(direction[1], 6)]
    else:
        goal = _resolve_goal(state, reference)
        if not goal:
            return {}
        goal_x, goal_y = goal["xy"]
        dx = goal_x - payload_x
        dy = goal_y - payload_y
        distance_m = hypot(dx, dy)
        if distance_m <= 1.0e-9:
            bearing_deg = heading_deg
            error_deg = 0.0
        else:
            bearing_deg = degrees(atan2(dy, dx))
            error_deg = wrap_degrees(bearing_deg - heading_deg)
        goal_source = goal["source"]
        reference_geometry = "world_point"
        goal_world_xy = [round(goal_x, 6), round(goal_y, 6)]
        goal_world_direction = []

    tolerance_deg = abs(float(heading_tolerance_deg))
    if not isfinite(tolerance_deg) or tolerance_deg <= 0.0:
        tolerance_deg = HEADING_ALIGNMENT_TOLERANCE_DEG
    abs_error_deg = abs(error_deg)
    aligned = abs_error_deg <= tolerance_deg
    allowed_turns = _allowed_turn_directions(error_deg, aligned=aligned)
    direction_scores = _body_direction_alignment_scores(error_deg)
    preferred_translation = max(
        direction_scores,
        key=direction_scores.__getitem__,
    )
    minimum_translation_alignment = cos(radians(tolerance_deg))
    aligned_translation_directions = [
        direction
        for direction, score in direction_scores.items()
        if score >= minimum_translation_alignment - 1.0e-9
    ]
    legal_translation_directions = _legal_translation_directions(
        reference=reference,
        aligned=aligned,
        geometry_aligned_directions=aligned_translation_directions,
    )
    checkpoint = state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    return {
        "schema_version": LOCAL_NAVIGATION_SCHEMA_VERSION,
        "kind": "local_heading_to_declared_reference",
        "active": True,
        "reference": reference,
        "reference_source": reference_source,
        "reference_geometry": reference_geometry,
        "goal_source": goal_source,
        "goal_world_xy_m": goal_world_xy,
        "goal_world_direction_xy": goal_world_direction,
        "declared_world_direction_xy": (
            [round(declared_direction[0], 6), round(declared_direction[1], 6)]
            if declared_direction
            else []
        ),
        "direction_locked_by_active_turn": bool(locked_direction),
        "route_reference_locked": bool(
            navigation.get("route_reference_locked", False)
        ),
        "validated_route_segment_id": str(
            navigation.get("validated_route_segment_id", "")
        ),
        "route_observation_captured_at_unix": navigation.get(
            "route_observation_captured_at_unix", 0.0
        ),
        "declared_direction_matches_locked": not direction_mismatch,
        "payload_world_xy_m": [round(payload_x, 6), round(payload_y, 6)],
        "goal_distance_m": (
            round(distance_m, 6) if distance_m is not None else None
        ),
        "team_heading_deg": round(heading_deg, 6),
        "goal_bearing_deg": round(wrap_degrees(bearing_deg), 6),
        "signed_heading_error_deg": round(error_deg, 6),
        "absolute_heading_error_deg": round(abs_error_deg, 6),
        "heading_tolerance_deg": tolerance_deg,
        "heading_aligned": aligned,
        "heading_alignment_required": not aligned,
        "required_turn_direction": (
            allowed_turns[0] if len(allowed_turns) == 1 else ""
        ),
        "allowed_turn_directions": list(allowed_turns),
        "body_translation_alignment": direction_scores,
        "preferred_translation_direction": preferred_translation,
        "translation_directions_within_tolerance": legal_translation_directions,
        "forward_translation_allowed": aligned,
        "minimum_heading_improvement_deg": MIN_HEADING_IMPROVEMENT_DEG,
        "scope_checkpoint_id": str(checkpoint.get("checkpoint_id", "")).strip(),
        "sign_convention": (
            "positive heading error requires counterclockwise yaw; negative "
            "heading error requires clockwise yaw"
        ),
        "contract": (
            "The route and reference are selected by Task Orchestrator. The host "
            "only verifies local heading progress from live poses. Continue a "
            "same-sign shared-heading turn while it reduces absolute heading "
            "error; do not score that turn as linear target-distance progress."
        ),
    }


def target_local_navigation_evidence(
    runtime_state: dict[str, Any] | None,
    *,
    heading_tolerance_deg: float = HEADING_ALIGNMENT_TOLERANCE_DEG,
) -> dict[str, Any]:
    """Project the live final target without requiring a persistent task plan."""

    synthetic_plan = {
        "global_plan": [
            {
                "phase_id": "target_approach",
                "abstract_goal": "approach the final target",
                "completion_evidence": "arrive at the target",
            },
            {
                "phase_id": "payload_placement",
                "abstract_goal": "place and release the payload",
                "completion_evidence": "payload supported by target",
            },
        ],
        "current_phase_id": "target_approach",
        "current_subtask": {
            "navigation": {
                "reference": FINAL_TARGET_REFERENCE,
                "arrival_condition": "arrive at final target",
            }
        },
    }
    return build_local_navigation_contract(
        synthetic_plan,
        runtime_state,
        heading_tolerance_deg=heading_tolerance_deg,
    )


def validate_strategy_against_local_navigation(
    *,
    method: str,
    objective: str,
    contract: dict[str, Any] | None,
    allow_formation_repair: bool = False,
    allow_route_recovery: bool = False,
    allow_turn_clearance_replan: bool = False,
    route_clearance_conflict_contract: dict[str, Any] | None = None,
) -> None:
    """Reject a maneuver that contradicts an active local heading contract."""

    local = contract if isinstance(contract, dict) else {}
    if not local.get("active", False):
        return
    method = str(method).strip()
    objective = str(objective).strip().lower()
    alignment_required = bool(local.get("heading_alignment_required", False))
    allowed_turns = {
        str(item).strip().lower()
        for item in local.get("allowed_turn_directions", ())
        if str(item).strip()
    }
    clearance_conflict = (
        route_clearance_conflict_contract
        if isinstance(route_clearance_conflict_contract, dict)
        else {}
    )
    conflict_active = bool(clearance_conflict.get("active", False))
    conflict_turns = {
        str(item).strip().lower()
        for item in clearance_conflict.get(
            "safe_heading_correction_turn_directions",
            (),
        )
        if str(item).strip()
    }
    conflict_translations = {
        str(item).strip().lower()
        for key in (
            "route_reentry_translation_directions",
            "clearance_setup_translation_directions",
        )
        for item in clearance_conflict.get(key, ())
        if str(item).strip()
    }
    if not bool(local.get("declared_direction_matches_locked", True)):
        raise ValueError(
            "the active shared-heading turn has locked world_direction_xy="
            f"{local.get('goal_world_direction_xy')}; preserve that local route "
            "reference until alignment or a verifier-rejected rollback"
        )
    if method in LINEAR_METHODS:
        translation_direction = _translation_direction(method, objective)
        aligned_translations = {
            str(item).strip().lower()
            for item in local.get("translation_directions_within_tolerance", ())
            if str(item).strip()
        }
        route_translation_allowed = translation_direction in aligned_translations
        route_clearance_reentry_allowed = bool(
            conflict_active and translation_direction in conflict_translations
        )
        recovery_authorized = bool(
            allow_formation_repair
            or allow_route_recovery
            or allow_turn_clearance_replan
            or route_clearance_reentry_allowed
        )
        if not route_translation_allowed and not recovery_authorized:
            reference = str(local.get("reference", "")).strip()
            if reference == VISIBLE_PASSAGE_REFERENCE:
                route_guidance = (
                    ", or execute a bounded shared-heading "
                    "coordinated_pivot with semantic_objective in "
                    f"{sorted(allowed_turns)}. "
                    if alignment_required
                    else ". Entering heading tolerance does not authorize "
                    "the opposite translation direction. "
                )
                raise ValueError(
                    "the selected body-frame translation does not follow the "
                    "declared visible-passage direction. Select one of the "
                    "geometry-consistent body-frame translations "
                    f"{sorted(aligned_translations)}"
                    f"{route_guidance}"
                    f"reference={reference!r}, "
                    "signed_heading_error_deg="
                    f"{local.get('signed_heading_error_deg')}"
                )
            if alignment_required:
                raise ValueError(
                    f"reference={reference!r} is a world-point navigation target and "
                    "heading_alignment_required=true. No body-frame linear "
                    "translation, including backward, substitutes for shared-heading "
                    "alignment. Execute a bounded coordinated_pivot with "
                    f"semantic_objective in {sorted(allowed_turns)} until the measured "
                    "heading enters tolerance, then replan translation. "
                    "signed_heading_error_deg="
                    f"{local.get('signed_heading_error_deg')}"
                )
            raise ValueError(
                f"body-frame translation {translation_direction!r} points away "
                f"from the active {reference!r} navigation reference. Select one "
                "of the live geometry-consistent directions "
                f"{sorted(aligned_translations)}; heading_aligned=true does not "
                "authorize an opposite route translation"
            )
    if (
        alignment_required
        and method == "coordinated_alignment"
        and not allow_formation_repair
    ):
        raise ValueError(
            "coordinated_alignment repairs front/back formation error but does "
            "not replace the shared-heading turn required by the declared local "
            "navigation reference"
        )
    route_clearance_turn_allowed = bool(
        conflict_active and objective in conflict_turns
    )
    if (
        method in TURN_METHODS
        and allowed_turns
        and objective not in allowed_turns
        and not route_clearance_turn_allowed
    ):
        raise ValueError(
            f"turn_direction={objective!r} increases the shortest local heading "
            f"error; choose one of {sorted(allowed_turns)} for reference="
            f"{local.get('reference')!r}"
        )
    if (
        method in TURN_METHODS
        and not alignment_required
        and not route_clearance_turn_allowed
    ):
        raise ValueError(
            "the shared team heading is already within the declared local "
            f"navigation tolerance for reference={local.get('reference')!r}; "
            "stop turning and select a translation or the next semantic subtask"
        )


def evaluate_local_navigation_transition(
    contract: dict[str, Any] | None,
    before_state: dict[str, Any] | None,
    after_state: dict[str, Any] | None,
    *,
    requested_turn_direction: str = "",
) -> dict[str, Any]:
    """Measure whether one rollout improved the declared local heading."""

    local = contract if isinstance(contract, dict) else {}
    before = before_state if isinstance(before_state, dict) else {}
    after = after_state if isinstance(after_state, dict) else {}
    before_error = _heading_error_to_reference(before, local)
    after_error = _heading_error_to_reference(after, local)
    if before_error is None or after_error is None:
        return {"available": False, "reason": "live_pose_missing"}
    tolerance = _finite_positive(
        local.get("heading_tolerance_deg"),
        HEADING_ALIGNMENT_TOLERANCE_DEG,
    )
    minimum_improvement = _finite_positive(
        local.get("minimum_heading_improvement_deg"),
        MIN_HEADING_IMPROVEMENT_DEG,
    )
    improvement = abs(before_error) - abs(after_error)
    before_heading = team_heading_deg(before)
    after_heading = team_heading_deg(after)
    heading_change = (
        wrap_degrees(after_heading - before_heading)
        if before_heading is not None and after_heading is not None
        else 0.0
    )
    aligned_before = abs(before_error) <= tolerance
    aligned_after = abs(after_error) <= tolerance
    allowed_before = set(_allowed_turn_directions(before_error, aligned=aligned_before))
    requested = str(requested_turn_direction).strip().lower()
    requested_sign = {
        "clockwise": -1.0,
        "counterclockwise": 1.0,
    }.get(requested, 0.0)
    heading_motion_matches_request = bool(
        not requested_sign
        or requested_sign * heading_change >= minimum_improvement
    )
    meaningful_heading_motion = abs(heading_change) >= minimum_improvement
    improved = bool(
        meaningful_heading_motion
        and heading_motion_matches_request
        and (aligned_after or improvement >= minimum_improvement)
    )
    return {
        "available": True,
        "reference": str(local.get("reference", "")),
        "signed_heading_error_before_deg": round(before_error, 6),
        "signed_heading_error_after_deg": round(after_error, 6),
        "absolute_heading_error_before_deg": round(abs(before_error), 6),
        "absolute_heading_error_after_deg": round(abs(after_error), 6),
        "heading_error_improvement_deg": round(improvement, 6),
        "team_heading_change_deg": round(heading_change, 6),
        "heading_tolerance_deg": tolerance,
        "heading_aligned_before": aligned_before,
        "heading_aligned_after": aligned_after,
        "minimum_heading_improvement_deg": minimum_improvement,
        "semantic_heading_improved": improved,
        "requested_turn_direction": requested,
        "allowed_turn_directions_before": sorted(allowed_before),
        "requested_turn_matches_local_error": bool(
            not requested or not allowed_before or requested in allowed_before
        ),
        "measured_heading_motion_matches_request": heading_motion_matches_request,
        "meaningful_team_heading_motion": meaningful_heading_motion,
        "target_distance_is_not_turn_success": True,
    }


def refresh_local_navigation_contract(
    contract: dict[str, Any] | None,
    runtime_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Refresh one declared reference against a newer trusted state.

    The reference and world goal remain those selected by the Orchestrator.
    Only pose-derived heading evidence is recomputed; no route action is
    selected here.
    """

    local = contract if isinstance(contract, dict) else {}
    state = runtime_state if isinstance(runtime_state, dict) else {}
    if not local.get("active", False):
        return {}
    heading_deg = team_heading_deg(state)
    payload_xy = _payload_xy(state)
    if heading_deg is None or payload_xy is None:
        return {}
    if str(local.get("reference_geometry", "")) == "world_direction":
        direction = _world_direction_xy(local.get("goal_world_direction_xy"))
        if direction is None:
            return {}
        bearing_deg = degrees(atan2(direction[1], direction[0]))
        distance_m: float | None = None
    else:
        goal_xy = local.get("goal_world_xy_m", ())
        try:
            goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
        except (IndexError, TypeError, ValueError):
            return {}
        dx = goal_x - payload_xy[0]
        dy = goal_y - payload_xy[1]
        distance_m = hypot(dx, dy)
        bearing_deg = (
            heading_deg if distance_m <= 1.0e-9 else degrees(atan2(dy, dx))
        )
    error_deg = wrap_degrees(bearing_deg - heading_deg)
    tolerance = _finite_positive(
        local.get("heading_tolerance_deg"),
        HEADING_ALIGNMENT_TOLERANCE_DEG,
    )
    aligned = abs(error_deg) <= tolerance
    allowed_turns = _allowed_turn_directions(error_deg, aligned=aligned)
    direction_scores = _body_direction_alignment_scores(error_deg)
    minimum_translation_alignment = cos(radians(tolerance))
    geometry_aligned_directions = [
        direction
        for direction, score in direction_scores.items()
        if score >= minimum_translation_alignment - 1.0e-9
    ]
    legal_translation_directions = _legal_translation_directions(
        reference=str(local.get("reference", "")).strip(),
        aligned=aligned,
        geometry_aligned_directions=geometry_aligned_directions,
    )
    return {
        **local,
        "goal_distance_m": (
            round(distance_m, 6) if distance_m is not None else None
        ),
        "payload_world_xy_m": [round(payload_xy[0], 6), round(payload_xy[1], 6)],
        "team_heading_deg": round(heading_deg, 6),
        "goal_bearing_deg": round(wrap_degrees(bearing_deg), 6),
        "signed_heading_error_deg": round(error_deg, 6),
        "absolute_heading_error_deg": round(abs(error_deg), 6),
        "heading_aligned": aligned,
        "heading_alignment_required": not aligned,
        "required_turn_direction": (
            allowed_turns[0] if len(allowed_turns) == 1 else ""
        ),
        "allowed_turn_directions": list(allowed_turns),
        "body_translation_alignment": direction_scores,
        "preferred_translation_direction": max(
            direction_scores,
            key=direction_scores.__getitem__,
        ),
        "translation_directions_within_tolerance": legal_translation_directions,
        "forward_translation_allowed": aligned,
    }


def _legal_translation_directions(
    *,
    reference: str,
    aligned: bool,
    geometry_aligned_directions: list[str],
) -> list[str]:
    """Expose translations that satisfy the reference semantics, not just geometry."""

    if reference == VISIBLE_PASSAGE_REFERENCE:
        return list(geometry_aligned_directions)
    if reference in {FINAL_TARGET_REFERENCE, RUNTIME_LOCAL_GOAL_REFERENCE}:
        return ["forward"] if aligned else []
    return []


def team_heading_deg(runtime_state: dict[str, Any] | None) -> float | None:
    """Return the circular mean of available front/back robot root yaw."""

    state = runtime_state if isinstance(runtime_state, dict) else {}
    robots = state.get("robots", {})
    if not isinstance(robots, dict):
        return None
    samples: list[float] = []
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            continue
        try:
            yaw = (
                float(robot["yaw_rad"])
                if robot.get("yaw_rad") is not None
                else radians(float(robot["yaw_deg"]))
            )
        except (KeyError, TypeError, ValueError):
            continue
        if isfinite(yaw):
            samples.append(yaw)
    if not samples:
        return None
    return wrap_degrees(
        degrees(
            atan2(
                sum(sin(value) for value in samples),
                sum(cos(value) for value in samples),
            )
        )
    )


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _navigation_reference(task_plan: dict[str, Any]) -> tuple[str, str]:
    navigation = _subtask_navigation(task_plan)
    explicit = str(navigation.get("reference", "")).strip().lower()
    if explicit:
        return explicit, "task_orchestrator.current_subtask.navigation"
    if _is_generic_target_approach_phase(task_plan):
        return FINAL_TARGET_REFERENCE, "generic_preplacement_phase_relation"
    return VISIBLE_PASSAGE_REFERENCE, "default_route_discovery_reference"


def _subtask_navigation(task_plan: dict[str, Any]) -> dict[str, Any]:
    subtask = task_plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    navigation = subtask.get("navigation", {})
    return navigation if isinstance(navigation, dict) else {}


def _world_direction_xy(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    magnitude = hypot(x, y)
    if not isfinite(magnitude) or magnitude <= 1.0e-6:
        return None
    return x / magnitude, y / magnitude


def _locked_navigation_direction(
    runtime_state: dict[str, Any],
    *,
    reference: str,
) -> tuple[float, float] | None:
    episode = runtime_state.get("turn_episode", {})
    episode = episode if isinstance(episode, dict) else {}
    if not (
        str(episode.get("purpose", "")) == "local_navigation_turn"
        and str(episode.get("reference", "")) == reference
        and (
            bool(episode.get("active", False))
            or bool(episode.get("translation_pending", False))
        )
    ):
        return None
    return _world_direction_xy(episode.get("world_direction_xy"))


def _direction_dot(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _is_generic_target_approach_phase(task_plan: dict[str, Any]) -> bool:
    phases = task_plan.get("global_plan", ())
    if not isinstance(phases, list) or not phases:
        return False
    current_id = str(task_plan.get("current_phase_id", "")).strip()
    current_index = next(
        (
            index
            for index, phase in enumerate(phases)
            if isinstance(phase, dict)
            and str(phase.get("phase_id", "")).strip() == current_id
        ),
        -1,
    )
    if current_index < 0:
        return False
    current_text = _phase_text(phases[current_index])
    if _contains_target_approach_terms(current_text):
        return True
    if current_index + 1 >= len(phases):
        return False
    return _contains_payload_lifecycle_terms(_phase_text(phases[current_index + 1]))


def _phase_text(phase: object) -> str:
    if not isinstance(phase, dict):
        return ""
    return " ".join(
        str(phase.get(key, "")).strip().lower()
        for key in ("phase_id", "abstract_goal", "completion_evidence")
    )


def _contains_target_approach_terms(text: str) -> bool:
    target = any(term in text for term in ("target", "goal", "destination", "pedestal"))
    approach = any(
        term in text
        for term in ("approach", "transport", "carry", "align", "straight")
    )
    lifecycle = _contains_payload_lifecycle_terms(text)
    return target and approach and not lifecycle


def _contains_payload_lifecycle_terms(text: str) -> bool:
    payload = any(term in text for term in ("payload", "box", "object"))
    lifecycle = any(
        term in text
        for term in ("place", "placement", "release", "deposit", "set down")
    )
    return payload and lifecycle


def _resolve_goal(
    runtime_state: dict[str, Any],
    reference: str,
) -> dict[str, Any]:
    if reference == FINAL_TARGET_REFERENCE:
        source = runtime_state.get("target", {})
        source_name = "runtime_state.target"
        x_keys = ("center_x_m", "x")
        y_keys = ("center_y_m", "y")
    else:
        source = runtime_state.get("local_navigation_goal", {})
        source_name = "runtime_state.local_navigation_goal"
        x_keys = ("x_m", "center_x_m", "x")
        y_keys = ("y_m", "center_y_m", "y")
    if not isinstance(source, dict):
        return {}
    x = _first_finite(source, x_keys)
    y = _first_finite(source, y_keys)
    if x is None or y is None:
        return {}
    return {"xy": (x, y), "source": source_name}


def _payload_xy(runtime_state: dict[str, Any]) -> tuple[float, float] | None:
    payload = runtime_state.get("payload", {})
    if not isinstance(payload, dict):
        return None
    try:
        x = float(payload["x"])
        y = float(payload["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y) if isfinite(x) and isfinite(y) else None


def _heading_error_to_goal(
    runtime_state: dict[str, Any],
    goal_xy: tuple[float, float],
) -> float | None:
    heading = team_heading_deg(runtime_state)
    payload_xy = _payload_xy(runtime_state)
    if heading is None or payload_xy is None:
        return None
    dx = goal_xy[0] - payload_xy[0]
    dy = goal_xy[1] - payload_xy[1]
    if hypot(dx, dy) <= 1.0e-9:
        return 0.0
    return wrap_degrees(degrees(atan2(dy, dx)) - heading)


def _heading_error_to_reference(
    runtime_state: dict[str, Any],
    contract: dict[str, Any],
) -> float | None:
    heading = team_heading_deg(runtime_state)
    if heading is None:
        return None
    if str(contract.get("reference_geometry", "")) == "world_direction":
        direction = _world_direction_xy(contract.get("goal_world_direction_xy"))
        if direction is None:
            return None
        return wrap_degrees(degrees(atan2(direction[1], direction[0])) - heading)
    goal_xy = contract.get("goal_world_xy_m", ())
    try:
        goal = (float(goal_xy[0]), float(goal_xy[1]))
    except (IndexError, TypeError, ValueError):
        return None
    return _heading_error_to_goal(runtime_state, goal)


def _allowed_turn_directions(error_deg: float, *, aligned: bool) -> tuple[str, ...]:
    if aligned:
        return ()
    if abs(abs(error_deg) - 180.0) <= AMBIGUOUS_REAR_HEADING_DEG:
        return ("clockwise", "counterclockwise")
    return ("counterclockwise",) if error_deg > 0.0 else ("clockwise",)


def _body_direction_alignment_scores(error_deg: float) -> dict[str, float]:
    offsets = {
        "forward": 0.0,
        "backward": 180.0,
        "left": 90.0,
        "right": -90.0,
    }
    return {
        direction: round(cos(radians(wrap_degrees(error_deg - offset))), 6)
        for direction, offset in offsets.items()
    }


def _translation_direction(method: str, objective: str) -> str:
    legacy = {
        "carry_forward": "forward",
        "carry_backward": "backward",
        "carry_lateral_left": "left",
        "carry_lateral_right": "right",
    }
    return legacy.get(str(method).strip(), str(objective).strip().lower())


def _first_finite(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        try:
            value = float(source[key])
        except (KeyError, TypeError, ValueError):
            continue
        if isfinite(value):
            return value
    return None


def _finite_positive(value: object, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if isfinite(parsed) and parsed > 0.0 else fallback


__all__ = [
    "FINAL_TARGET_REFERENCE",
    "FORMATION_REFERENCE",
    "HEADING_ALIGNMENT_TOLERANCE_DEG",
    "LINEAR_METHODS",
    "LOCAL_NAVIGATION_CONTRACT_KEY",
    "MIN_HEADING_IMPROVEMENT_DEG",
    "PAYLOAD_LIFECYCLE_REFERENCE",
    "RUNTIME_LOCAL_GOAL_REFERENCE",
    "TURN_METHODS",
    "VISIBLE_PASSAGE_REFERENCE",
    "build_local_navigation_contract",
    "evaluate_local_navigation_transition",
    "refresh_local_navigation_contract",
    "team_heading_deg",
    "target_local_navigation_evidence",
    "validate_strategy_against_local_navigation",
    "wrap_degrees",
]
