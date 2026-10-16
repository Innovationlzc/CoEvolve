"""Persistent signed-turn constraints for loaded route discovery."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .contracts import EvaluationReport, PolicyScript
from .coordination_specs import (
    COORDINATED_ALIGNMENT,
    COORDINATED_ARC_TRANSPORT,
    COORDINATED_LINEAR_TRANSPORT,
    COORDINATED_PIVOT,
    MODEL_ROUTE_TURN_ANGLE_DEG,
)

if TYPE_CHECKING:
    from .visual_verifier import VisualVerifierReport


STATIC_TURN_DIRECTIONS = {
    "turn_clockwise": "clockwise",
    "turn_counterclockwise": "counterclockwise",
}
STATIC_TURN_METHODS = frozenset(STATIC_TURN_DIRECTIONS)
ROLE_BASED_TURN_METHODS = frozenset(
    {COORDINATED_ARC_TRANSPORT, COORDINATED_PIVOT}
)
# Every loaded shared-heading implementation participates in the same
# world-frame direction lock and stop condition.  The route contract must not
# disappear merely because the Coder selects the static synchronized turn
# instead of an inner/outer coordination primitive.
TURN_METHODS = ROLE_BASED_TURN_METHODS | STATIC_TURN_METHODS
TURN_DIRECTIONS = frozenset({"clockwise", "counterclockwise"})
PAYLOAD_LIFECYCLE_METHODS = frozenset(
    {
        "place_crouch",
        "release_payload",
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
        "recover_stand",
    }
)
TURN_EPISODE_SCHEMA_VERSION = 3
MAX_ALIGNMENT_PROBE_DISTANCE_M = 1.0
ROUTE_TURN_PROBE_YAW_DEG = 85.0
LOCAL_NAVIGATION_TURN_BUDGET_DEG = 120.0
PHYSICS_PASSAGE_AXIS_ALIGNMENT_MIN = 0.85
PHYSICS_ROBOT_HEADING_SPLIT_MAX_DEG = 15.0
LONGITUDINAL_ALIGNMENT_COSINE_MIN = 2.0**-0.5
HEADING_CORRECTION_PURPOSE = "heading_correction"
ROUTE_TURN_PURPOSE = "route_turn"
PASSAGE_ALIGNMENT_REQUIRED_STATUS = "passage_alignment_required"
PASSAGE_ALIGNED_STATUS = "passage_aligned"
HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE = "host_wall_clearance_escape"
HOST_WALL_CLEARANCE_ESCAPE_PARAMETER_SOURCE = (
    "host_wall_clearance_escape_fallback"
)
HOST_WALL_CLEARANCE_ESCAPE_MARKER = "[host_wall_clearance_escape]"
HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE = "host_wall_turn_escape"
HOST_WALL_TURN_ESCAPE_PARAMETER_SOURCE = (
    "host_wall_turn_escape_model_parameters"
)
HOST_WALL_TURN_ESCAPE_MARKER = "[host_wall_turn_escape]"
HOST_WALL_HEADING_CORRECTION_SELECTION_SOURCE = "host_wall_heading_correction"
HOST_WALL_HEADING_CORRECTION_PARAMETER_SOURCE = (
    "host_wall_heading_correction_fallback"
)
HOST_WALL_HEADING_CORRECTION_MARKER = "[host_wall_heading_correction]"
ALIGNED_PASSAGE_POSTURES = frozenset(
    {"longitudinal_aligned", "lateral_aligned"}
)
TURN_SCRIPT_REWRITE_FAILURES = frozenset(
    {
        "payload_drop",
        "payload_not_supported",
        "payload_slip",
        "payload_unstable",
        "formation_desync",
        "robot_pair_too_close",
        "robot_fall",
    }
)


class TurnEpisodeError(ValueError):
    """Raised when a policy violates an active turn episode."""


@dataclass(frozen=True)
class TurnIntent:
    method: str
    direction: str
    roles: tuple[tuple[str, str], ...]
    angle_deg: float


def empty_turn_episode() -> dict[str, Any]:
    return {
        "schema_version": TURN_EPISODE_SCHEMA_VERSION,
        "active": False,
        "status": "inactive",
        "purpose": "",
        "direction": "",
        "roles": {},
        "accepted_turn_steps": 0,
        "switch_allowed": False,
        "probe_direction": "forward",
        "probe_max_distance_m": MAX_ALIGNMENT_PROBE_DISTANCE_M,
    }


def normalize_turn_episode(
    episode: dict[str, Any] | None,
) -> dict[str, Any]:
    """Migrate a persisted turn episode and enforce its physical turn budget."""

    current = dict(episode) if isinstance(episode, dict) else empty_turn_episode()
    current["schema_version"] = TURN_EPISODE_SCHEMA_VERSION
    if str(current.get("purpose", "")) == ROUTE_TURN_PURPOSE:
        status = str(current.get("status", ""))
        if status in {"completed", "probe_ready"} and bool(
            current.get("translation_pending", False)
        ):
            if _episode_has_verified_passage_alignment(current):
                current.update(
                    {
                        "active": False,
                        "status": PASSAGE_ALIGNED_STATUS,
                        "switch_allowed": False,
                        "translation_pending": True,
                    }
                )
            else:
                current.update(
                    {
                        "active": False,
                        "status": PASSAGE_ALIGNMENT_REQUIRED_STATUS,
                        "switch_allowed": True,
                        "translation_pending": False,
                        "completion_trigger": (
                            "turn_budget_reached_without_passage_alignment"
                        ),
                    }
                )
    if (
        not bool(current.get("active", False))
        or str(current.get("purpose", ""))
        not in {ROUTE_TURN_PURPOSE, "local_navigation_turn"}
    ):
        return current

    if (
        str(current.get("purpose", "")) == "local_navigation_turn"
        and str(current.get("status", "turning")) == "turning"
        and _float(current.get("cumulative_turn_progress_deg"))
        >= LOCAL_NAVIGATION_TURN_BUDGET_DEG
    ):
        current.update(
            {
                "active": False,
                "status": "local_direction_replan_required",
                "switch_allowed": True,
                "implementation_switch_allowed": True,
                "translation_pending": False,
                "completion_trigger": "local_turn_budget_reached",
                "rewrite_requirement": (
                    "capture a fresh global observation and validate a new route "
                    "reference before another shared-heading turn"
                ),
            }
        )
        return current

    if "cumulative_signed_payload_yaw_deg" not in current:
        accepted_steps = max(0, int(current.get("accepted_turn_steps", 0)))
        nominal_angle_deg = max(0.0, _float(current.get("last_angle_deg")))
        estimated_signed_yaw_deg = _turn_direction_sign(
            str(current.get("direction", ""))
        ) * accepted_steps * nominal_angle_deg
        current["cumulative_signed_payload_yaw_deg"] = estimated_signed_yaw_deg
        current["cumulative_payload_turn_progress_deg"] = abs(
            estimated_signed_yaw_deg
        )
        current["cumulative_turn_progress_deg"] = abs(estimated_signed_yaw_deg)
        current["cumulative_turn_progress_source"] = "legacy_nominal_fallback"
    elif "cumulative_payload_turn_progress_deg" not in current:
        direction_sign = _turn_direction_sign(str(current.get("direction", "")))
        current["cumulative_payload_turn_progress_deg"] = max(
            0.0,
            direction_sign * _float(
                current.get("cumulative_signed_payload_yaw_deg")
            ),
        )
        current["cumulative_turn_progress_deg"] = current[
            "cumulative_payload_turn_progress_deg"
        ]
        current["cumulative_turn_progress_source"] = "physical_payload_yaw"

    if "cumulative_signed_team_yaw_deg" not in current:
        accepted_steps = max(0, int(current.get("accepted_turn_steps", 0)))
        nominal_angle_deg = max(0.0, _float(current.get("last_angle_deg")))
        estimated_progress_deg = accepted_steps * nominal_angle_deg
        direction_sign = _turn_direction_sign(str(current.get("direction", "")))
        current["cumulative_signed_team_yaw_deg"] = (
            direction_sign * estimated_progress_deg
        )

    if (
        str(current.get("status", "turning")) == "turning"
        and str(current.get("purpose", "")) == ROUTE_TURN_PURPOSE
        and _float(current.get("cumulative_payload_turn_progress_deg"))
        >= ROUTE_TURN_PROBE_YAW_DEG
    ):
        if _episode_has_verified_passage_alignment(current):
            current.update(
                {
                    "active": False,
                    "status": PASSAGE_ALIGNED_STATUS,
                    "switch_allowed": False,
                    "translation_pending": True,
                    "completion_trigger": "global_visual_passage_alignment",
                }
            )
        else:
            current.update(
                {
                    "active": False,
                    "status": PASSAGE_ALIGNMENT_REQUIRED_STATUS,
                    "switch_allowed": True,
                    "translation_pending": False,
                    "completion_trigger": (
                        "turn_budget_reached_without_passage_alignment"
                    ),
                }
            )
    return current


def policy_turn_intent(policy: PolicyScript) -> TurnIntent | None:
    calls = _policy_calls(policy)
    methods = {method for method, _arguments in calls.values()}
    if len(methods) != 1 or next(iter(methods), "") not in TURN_METHODS:
        return None
    method = next(iter(methods))
    directions = {
        (
            STATIC_TURN_DIRECTIONS[method]
            if method in STATIC_TURN_METHODS
            else str(arguments.get("turn_direction", "")).strip().lower()
        )
        for _method, arguments in calls.values()
    }
    if len(directions) != 1 or next(iter(directions), "") not in TURN_DIRECTIONS:
        return None
    direction = next(iter(directions))
    angle_parameter = "yaw_deg" if method in STATIC_TURN_METHODS else "angle_deg"
    angles = {
        abs(_float(arguments.get(angle_parameter)))
        for _method, arguments in calls.values()
    }
    if len(angles) != 1 or next(iter(angles), 0.0) <= 0.0:
        return None
    angle_deg = next(iter(angles))
    roles: dict[str, str] = {}
    if method in STATIC_TURN_METHODS:
        # A static turn is one identical synchronized team transaction.  It has
        # no temporary inner/outer role relation to preserve.
        roles = {}
    elif method == COORDINATED_PIVOT:
        roles = {
            robot_id: str(arguments.get("role", "")).strip().lower()
            for robot_id, (_method, arguments) in calls.items()
        }
    else:
        radii = {
            robot_id: _float(arguments.get("radius_m"))
            for robot_id, (_method, arguments) in calls.items()
        }
        if set(radii) == {"front", "back"} and radii["front"] != radii["back"]:
            roles = {
                "front": "outer" if radii["front"] > radii["back"] else "inner",
                "back": "outer" if radii["back"] > radii["front"] else "inner",
            }
    if method in ROLE_BASED_TURN_METHODS and (
        set(roles) != {"front", "back"}
        or set(roles.values()) != {"inner", "outer"}
    ):
        return None
    return TurnIntent(
        method=method,
        direction=direction,
        roles=tuple(sorted(roles.items())),
        angle_deg=angle_deg,
    )


def policy_linear_probe_distance(policy: PolicyScript) -> float | None:
    calls = _policy_calls(policy)
    if set(calls) != {"front", "back"}:
        return None
    if {method for method, _arguments in calls.values()} != {
        COORDINATED_LINEAR_TRANSPORT
    }:
        return None
    distances = {
        _float(arguments.get("distance_m"))
        for _method, arguments in calls.values()
    }
    if len(distances) != 1:
        return None
    return next(iter(distances))


def policy_linear_probe_direction(policy: PolicyScript) -> str:
    calls = _policy_calls(policy)
    directions = {
        str(arguments.get("direction", "")).strip().lower()
        for method, arguments in calls.values()
        if method == COORDINATED_LINEAR_TRANSPORT
    }
    return next(iter(directions)) if len(directions) == 1 else ""


def policy_yaw_alignment_deltas(
    policy: PolicyScript,
) -> dict[str, float]:
    calls = _policy_calls(policy)
    if set(calls) != {"front", "back"}:
        return {}
    if {method for method, _arguments in calls.values()} != {
        COORDINATED_ALIGNMENT
    }:
        return {}
    if {
        str(arguments.get("axis", "")).strip().lower()
        for _method, arguments in calls.values()
    } != {"yaw"}:
        return {}
    deltas = {
        robot_id: _float(arguments.get("delta"))
        for robot_id, (_method, arguments) in calls.items()
    }
    return deltas if abs(deltas["front"] - deltas["back"]) > 1.0e-9 else {}


def validate_policy_for_turn_episode(
    policy: PolicyScript,
    episode: dict[str, Any] | None,
    *,
    allow_final_target_reorientation: bool = False,
    allow_route_clearance_repair: bool = False,
) -> None:
    episode = episode if isinstance(episode, dict) else {}
    if str(episode.get("status", "")) == "local_direction_replan_required":
        # The old world direction is deliberately released. The next policy must
        # carry a freshly route-validated local-navigation contract.
        return
    intent = policy_turn_intent(policy)
    policy_methods = {method for method, _arguments in _policy_calls(policy).values()}
    if len(policy_methods) == 1 and next(iter(policy_methods)) in PAYLOAD_LIFECYCLE_METHODS:
        # Lifecycle availability is derived from current trusted physics by a
        # separate hard gate. Once that gate opens, stale route-turn continuity
        # cannot override placement, release, retreat, or upright recovery.
        return
    probe_distance = policy_linear_probe_distance(policy)
    wall_clearance_escape = bool(
        probe_distance is not None
        and HOST_WALL_CLEARANCE_ESCAPE_MARKER in policy.rationale
    )
    if (
        str(episode.get("purpose", "")) == ROUTE_TURN_PURPOSE
        and str(episode.get("status", "")) == PASSAGE_ALIGNMENT_REQUIRED_STATUS
    ):
        if intent is None and not wall_clearance_escape:
            raise TurnEpisodeError(
                "loaded straight transport is blocked because the current global "
                "view shows a diagonal or uncertain team-payload posture; first use "
                "a signed shared-heading turn to align longitudinally or laterally "
                "with a visible open passage, unless authored-wall geometry proves "
                "both turn headings blocked and marks an open translation as a "
                "clearance escape"
            )
        return
    small_heading_correction = bool(
        intent is not None
        and intent.angle_deg < MODEL_ROUTE_TURN_ANGLE_DEG - 1.0e-9
        and HOST_WALL_HEADING_CORRECTION_MARKER in policy.rationale
    )
    host_wall_turn_escape = bool(
        intent is not None
        and HOST_WALL_TURN_ESCAPE_MARKER in policy.rationale
    )
    if wall_clearance_escape or host_wall_turn_escape:
        # Exact current wall geometry authorized this one replan. Ordinary
        # model-generated turns and translations still obey the episode lock.
        return
    if (
        bool(episode.get("translation_pending", False))
        and intent is not None
        and not small_heading_correction
        and not host_wall_turn_escape
        and not allow_final_target_reorientation
        and not allow_route_clearance_repair
    ):
        raise TurnEpisodeError(
            "the global view already verifies a longitudinal or lateral passage-aligned "
            "posture; select a loaded linear direction from current evidence before "
            "starting another shared-heading turn"
        )
    if not bool(episode.get("active", False)):
        return
    if str(episode.get("purpose", "")) == "local_navigation_turn":
        if intent is None:
            if allow_route_clearance_repair:
                # Pre-dispatch has already checked the checkpoint-scoped wall
                # contract, sync mode, and exact open translation direction.
                # This setup move preserves the unfinished signed turn instead
                # of pretending that local heading alignment has completed.
                return
            raise TurnEpisodeError(
                "the declared local heading remains unresolved; continue the "
                "same signed coordinated turn before linear transport"
            )
        expected_direction = str(episode.get("direction", ""))
        if intent.direction != expected_direction:
            raise TurnEpisodeError(
                "unfinished local-navigation turn requires turn_direction="
                f"{expected_direction!r}"
            )
        implementation_switch_allowed = bool(
            episode.get("implementation_switch_allowed", False)
        )
        expected_method = str(episode.get("primitive", ""))
        if (
            not implementation_switch_allowed
            and expected_method
            and intent.method != expected_method
        ):
            raise TurnEpisodeError(
                "unfinished local-navigation turn must preserve its accepted "
                f"implementation api.{expected_method}; got api.{intent.method}"
            )
        expected_roles = _episode_roles(episode)
        if (
            not implementation_switch_allowed
            and expected_roles
            and dict(intent.roles) != expected_roles
        ):
            raise TurnEpisodeError(
                "unfinished local-navigation turn must preserve accepted inner/"
                f"outer roles; expected {expected_roles}, got {dict(intent.roles)}"
            )
        return
    if str(episode.get("purpose", "")) == HEADING_CORRECTION_PURPOSE:
        expected_method = str(episode.get("primitive", COORDINATED_PIVOT))
        if intent is None or intent.method != expected_method:
            raise TurnEpisodeError(
                "straight-heading correction accepts only the verifier-required "
                f"api.{expected_method} transaction"
            )
    if intent is not None:
        if (
            str(episode.get("purpose", "")) == ROUTE_TURN_PURPOSE
            and str(episode.get("status", "")) == "probe_ready"
        ):
            raise TurnEpisodeError(
                "route-turn budget is complete; execute the bounded linear probe "
                "instead of adding another turn"
            )
        if bool(episode.get("switch_allowed", False)):
            return
        expected_direction = str(episode.get("direction", ""))
        expected_roles = _episode_roles(episode)
        if intent.direction != expected_direction:
            raise TurnEpisodeError(
                "unfinished turn episode requires turn_direction="
                f"{expected_direction!r}; the opposite direction is allowed only after the "
                "current direction is verifier-rejected"
            )
        if dict(intent.roles) != expected_roles:
            raise TurnEpisodeError(
                "unfinished turn episode must preserve the accepted inner/outer robot roles; "
                f"expected {expected_roles}, got {dict(intent.roles)}"
            )
        if str(episode.get("purpose", "")) == HEADING_CORRECTION_PURPOSE:
            expected_angle = _float(episode.get("correction_angle_deg"))
            if expected_angle <= 0.0 or abs(intent.angle_deg - expected_angle) > 1.0e-6:
                raise TurnEpisodeError(
                    "straight-heading correction must use the verifier-reported "
                    f"angle_deg={expected_angle:g}; got {intent.angle_deg:g}"
                )
        return

    if probe_distance is not None:
        if str(episode.get("status", "")) != "probe_ready":
            raise TurnEpisodeError(
                "loaded linear transport is blocked while the signed turn episode is unfinished; "
                "the global VLM must first report aligned_with_open_passage"
            )
        maximum = _float(
            episode.get("probe_max_distance_m", MAX_ALIGNMENT_PROBE_DISTANCE_M)
        )
        if probe_distance <= 0.0 or probe_distance > maximum + 1.0e-9:
            raise TurnEpisodeError(
                f"turn-alignment probe distance must be in (0, {maximum:g}] m, "
                f"got {probe_distance:g}"
            )
        expected_direction = str(
            episode.get("probe_direction", "forward")
        ).strip().lower()
        actual_direction = policy_linear_probe_direction(policy)
        if expected_direction and actual_direction != expected_direction:
            raise TurnEpisodeError(
                "turn-alignment probe must move along the verified post-turn "
                f"heading with direction={expected_direction!r}; got "
                f"{actual_direction!r}"
            )


def validate_selected_parameters_for_turn_episode(
    selected: dict[str, Any],
    episode: dict[str, Any] | None,
) -> None:
    episode = episode if isinstance(episode, dict) else {}
    if str(episode.get("status", "")) == "local_direction_replan_required":
        return
    method = str(selected.get("method", ""))
    wall_clearance_escape = bool(
        method == COORDINATED_LINEAR_TRANSPORT
        and str(selected.get("selection_source", ""))
        == HOST_WALL_CLEARANCE_ESCAPE_PARAMETER_SOURCE
    )
    wall_turn_escape = bool(
        method in TURN_METHODS
        and str(selected.get("selection_source", ""))
        == HOST_WALL_TURN_ESCAPE_PARAMETER_SOURCE
    )
    wall_heading_correction = bool(
        method == COORDINATED_PIVOT
        and str(selected.get("selection_source", ""))
        == HOST_WALL_HEADING_CORRECTION_PARAMETER_SOURCE
    )
    if wall_clearance_escape or wall_turn_escape or wall_heading_correction:
        return
    if (
        str(episode.get("purpose", "")) == ROUTE_TURN_PURPOSE
        and str(episode.get("status", "")) == PASSAGE_ALIGNMENT_REQUIRED_STATUS
    ):
        if method not in TURN_METHODS and not wall_clearance_escape:
            raise TurnEpisodeError(
                "loaded straight transport is unavailable until a signed "
                "shared-heading turn aligns the team-payload posture with the "
                "visible open passage, unless the host wall-clearance gate marks "
                "an open translation as the only locally feasible escape"
            )
        return
    if (
        bool(episode.get("translation_pending", False))
        and method in TURN_METHODS
        and not wall_turn_escape
    ):
        raise TurnEpisodeError(
            "the payload already completed this route turn; select linear transport "
            "or a bounded alignment before another turn"
        )
    if not bool(episode.get("active", False)):
        return
    robot_parameters = selected.get("robot_parameters", {})
    robot_parameters = robot_parameters if isinstance(robot_parameters, dict) else {}
    if str(episode.get("purpose", "")) == HEADING_CORRECTION_PURPOSE:
        expected_method = str(episode.get("primitive", COORDINATED_PIVOT))
        if method != expected_method:
            raise TurnEpisodeError(
                "straight-heading correction accepts only the verifier-required "
                f"api.{expected_method} parameters"
            )
    if (
        method in TURN_METHODS
        and str(episode.get("purpose", "")) == ROUTE_TURN_PURPOSE
        and str(episode.get("status", "")) == "probe_ready"
    ):
        raise TurnEpisodeError(
            "route-turn budget is complete; select the bounded linear probe"
        )
    if method in TURN_METHODS and not bool(episode.get("switch_allowed", False)):
        expected_direction = str(episode.get("direction", ""))
        directions = (
            {STATIC_TURN_DIRECTIONS[method]}
            if method in STATIC_TURN_METHODS
            else {
                str(parameters.get("turn_direction", ""))
                for parameters in robot_parameters.values()
                if isinstance(parameters, dict)
            }
        )
        if directions != {expected_direction}:
            raise TurnEpisodeError(
                f"turn episode requires turn_direction={expected_direction!r}"
            )
        actual_roles = _selected_roles(method, robot_parameters)
        expected_roles = _episode_roles(episode)
        implementation_switch_allowed = bool(
            episode.get("implementation_switch_allowed", False)
        )
        expected_method = str(episode.get("primitive", ""))
        if (
            str(episode.get("purpose", "")) == "local_navigation_turn"
            and not implementation_switch_allowed
            and expected_method
            and method != expected_method
        ):
            raise TurnEpisodeError(
                "turn episode requires the accepted implementation "
                f"api.{expected_method}; got api.{method}"
            )
        if (
            not implementation_switch_allowed
            and expected_roles
            and actual_roles != expected_roles
        ):
            raise TurnEpisodeError(
                "turn episode requires preserved inner/outer roles: "
                f"expected {expected_roles}, got {actual_roles}"
            )
        if str(episode.get("purpose", "")) == HEADING_CORRECTION_PURPOSE:
            expected_angle = _float(episode.get("correction_angle_deg"))
            angles = {
                _float(parameters.get("angle_deg"))
                for parameters in robot_parameters.values()
                if isinstance(parameters, dict)
            }
            if angles != {expected_angle}:
                raise TurnEpisodeError(
                    "straight-heading correction must use the verifier-reported "
                    f"angle_deg={expected_angle:g}"
                )
    if method == COORDINATED_LINEAR_TRANSPORT:
        maximum = _float(
            episode.get("probe_max_distance_m", MAX_ALIGNMENT_PROBE_DISTANCE_M)
        )
        distances = {
            _float(parameters.get("distance_m"))
            for parameters in robot_parameters.values()
            if isinstance(parameters, dict)
        }
        if str(episode.get("status", "")) != "probe_ready" or not distances:
            raise TurnEpisodeError("linear probe is unavailable before visual passage alignment")
        if max(distances) > maximum + 1.0e-9:
            raise TurnEpisodeError(
                f"turn-alignment probe distance must not exceed {maximum:g} m"
            )
        expected_direction = str(
            episode.get("probe_direction", "forward")
        ).strip().lower()
        directions = {
            str(parameters.get("direction", "")).strip().lower()
            for parameters in robot_parameters.values()
            if isinstance(parameters, dict)
        }
        if expected_direction and directions != {expected_direction}:
            raise TurnEpisodeError(
                "turn-alignment probe requires direction="
                f"{expected_direction!r}"
            )


def update_turn_episode(
    episode: dict[str, Any] | None,
    *,
    policy: PolicyScript,
    report: EvaluationReport,
    visual_report: VisualVerifierReport | None,
    checkpoint_id: str,
) -> dict[str, Any]:
    current = normalize_turn_episode(episode)
    correction_episode = _straight_heading_correction_episode(
        report,
        checkpoint_id=checkpoint_id,
    )
    if correction_episode is not None:
        return correction_episode
    intent = policy_turn_intent(policy)
    if intent is not None:
        local_navigation = (
            policy.verification_contract.get("local_navigation", {})
            if isinstance(policy.verification_contract, dict)
            else {}
        )
        local_navigation = (
            local_navigation if isinstance(local_navigation, dict) else {}
        )
        local_transition = (
            report.evidence.get("local_navigation_transition", {})
            if isinstance(report.evidence, dict)
            else {}
        )
        local_transition = (
            local_transition if isinstance(local_transition, dict) else {}
        )
        if report.success:
            if (
                local_navigation.get("active", False)
                and local_navigation.get("reference")
                in {
                    "final_target",
                    "runtime_local_goal",
                    "visible_connected_free_space",
                }
                and local_transition.get("available", False)
            ):
                aligned = bool(local_transition.get("heading_aligned_after", False))
                previous_steps = (
                    int(current.get("accepted_turn_steps", 0))
                    if str(current.get("purpose", "")) == "local_navigation_turn"
                    and str(current.get("direction", "")) == intent.direction
                    else 0
                )
                same_local_episode = bool(
                    str(current.get("purpose", "")) == "local_navigation_turn"
                    and str(current.get("direction", "")) == intent.direction
                    and list(current.get("world_direction_xy", ()))
                    == list(local_navigation.get("goal_world_direction_xy", ()))
                )
                previous_signed_team_yaw_deg = (
                    _float(current.get("cumulative_signed_team_yaw_deg"))
                    if same_local_episode
                    else 0.0
                )
                previous_signed_payload_yaw_deg = (
                    _float(current.get("cumulative_signed_payload_yaw_deg"))
                    if same_local_episode
                    else 0.0
                )
                previous_turn_progress_deg = (
                    _float(current.get("cumulative_turn_progress_deg"))
                    if same_local_episode
                    else 0.0
                )
                observed_team_yaw_deg = _float(
                    report.metrics.get("team_step_signed_yaw_change_deg")
                )
                observed_payload_yaw_deg = _float(
                    report.metrics.get("payload_step_signed_yaw_change_deg")
                )
                direction_sign = _turn_direction_sign(intent.direction)
                cumulative_signed_team_yaw_deg = (
                    previous_signed_team_yaw_deg + observed_team_yaw_deg
                )
                cumulative_signed_payload_yaw_deg = (
                    previous_signed_payload_yaw_deg + observed_payload_yaw_deg
                )
                cumulative_turn_progress_deg = (
                    previous_turn_progress_deg
                    + max(0.0, direction_sign * observed_team_yaw_deg)
                )
                budget_reached = (
                    cumulative_turn_progress_deg
                    >= LOCAL_NAVIGATION_TURN_BUDGET_DEG
                )
                requires_replan = budget_reached and not aligned
                return {
                    "schema_version": TURN_EPISODE_SCHEMA_VERSION,
                    "active": not aligned and not requires_replan,
                    "status": (
                        "local_heading_aligned"
                        if aligned
                        else "local_direction_replan_required"
                        if requires_replan
                        else "turning"
                    ),
                    "purpose": "local_navigation_turn",
                    "reference": str(local_navigation.get("reference", "")),
                    "reference_geometry": str(
                        local_navigation.get("reference_geometry", "")
                    ),
                    "world_direction_xy": list(
                        local_navigation.get("goal_world_direction_xy", ())
                    ),
                    "goal_world_xy_m": list(
                        local_navigation.get("goal_world_xy_m", ())
                    ),
                    "direction": intent.direction,
                    "roles": dict(intent.roles),
                    "primitive": intent.method,
                    "last_angle_deg": intent.angle_deg,
                    "accepted_turn_steps": previous_steps + 1,
                    "switch_allowed": requires_replan,
                    "implementation_switch_allowed": requires_replan,
                    # Keep the declared local direction locked through the
                    # first accepted translation. Alignment alone is a setup
                    # result, not evidence that the passage was traversed.
                    "translation_pending": aligned,
                    "last_observed_signed_team_yaw_deg": observed_team_yaw_deg,
                    "last_observed_signed_payload_yaw_deg": (
                        observed_payload_yaw_deg
                    ),
                    "cumulative_signed_team_yaw_deg": (
                        cumulative_signed_team_yaw_deg
                    ),
                    "cumulative_signed_payload_yaw_deg": (
                        cumulative_signed_payload_yaw_deg
                    ),
                    "cumulative_turn_progress_deg": cumulative_turn_progress_deg,
                    "cumulative_turn_progress_source": "physical_team_yaw",
                    "turn_budget_deg": LOCAL_NAVIGATION_TURN_BUDGET_DEG,
                    "route_observation_captured_at_unix": _float(
                        local_navigation.get(
                            "route_observation_captured_at_unix",
                            current.get("route_observation_captured_at_unix", 0.0),
                        )
                    ),
                    "replan_requires_fresh_global_observation": requires_replan,
                    "latest_checkpoint_id": checkpoint_id,
                    "completion_checkpoint_id": checkpoint_id if aligned else "",
                    "signed_heading_error_before_deg": local_transition.get(
                        "signed_heading_error_before_deg"
                    ),
                    "signed_heading_error_after_deg": local_transition.get(
                        "signed_heading_error_after_deg"
                    ),
                    "heading_error_improvement_deg": local_transition.get(
                        "heading_error_improvement_deg"
                    ),
                    "heading_tolerance_deg": local_transition.get(
                        "heading_tolerance_deg"
                    ),
                    "completion_trigger": (
                        "declared_local_heading_aligned"
                        if aligned
                        else "local_turn_budget_reached"
                        if requires_replan
                        else ""
                    ),
                    "rewrite_requirement": (
                        "capture a fresh global observation and validate a new "
                        "route reference before another shared-heading turn"
                        if requires_replan
                        else ""
                    ),
                }
            current_status = str(current.get("status", ""))
            current_purpose = str(current.get("purpose", ""))
            route_alignment_continuation = bool(
                current_purpose == ROUTE_TURN_PURPOSE
                and (
                    bool(current.get("active", False))
                    or current_status == PASSAGE_ALIGNMENT_REQUIRED_STATUS
                )
            )
            previous_steps = (
                int(current.get("accepted_turn_steps", 0))
                if (
                    bool(current.get("active", False))
                    or route_alignment_continuation
                )
                else 0
            )
            if (
                bool(current.get("active", False))
                and current_purpose == HEADING_CORRECTION_PURPOSE
            ):
                completed = dict(current)
                completed.update(
                    {
                        "active": False,
                        "status": "correction_completed",
                        "accepted_turn_steps": previous_steps + 1,
                        "switch_allowed": False,
                        "latest_checkpoint_id": checkpoint_id,
                    }
                )
                return completed
            visual_aligned = _visually_aligned_with_open_passage(
                visual_report
            )
            physics_posture = _physics_aligned_passage_posture(
                report,
                visual_report=visual_report,
            )
            aligned = visual_aligned or bool(physics_posture)
            passage_posture = (
                _visual_passage_posture(visual_report)
                if visual_aligned
                else physics_posture
            )
            alignment_source = (
                "global_visual_passage_alignment"
                if visual_aligned
                else "physics_active_goal_xy_alignment"
                if physics_posture
                else ""
            )
            same_route_episode = bool(
                route_alignment_continuation
                and (
                    (
                        str(current.get("direction", "")) == intent.direction
                        and _episode_roles(current) == dict(intent.roles)
                    )
                    or (
                        current_status == PASSAGE_ALIGNMENT_REQUIRED_STATUS
                        and bool(current.get("switch_allowed", False))
                    )
                )
            )
            previous_signed_yaw_deg = (
                _float(current.get("cumulative_signed_team_yaw_deg"))
                if same_route_episode
                else 0.0
            )
            previous_signed_payload_yaw_deg = (
                _float(current.get("cumulative_signed_payload_yaw_deg"))
                if same_route_episode
                else 0.0
            )
            previous_payload_progress_deg = (
                _float(current.get("cumulative_payload_turn_progress_deg"))
                if same_route_episode
                else 0.0
            )
            observed_signed_team_yaw_deg = _float(
                report.metrics.get("team_step_signed_yaw_change_deg")
            )
            observed_signed_payload_yaw_deg = _float(
                report.metrics.get("payload_step_signed_yaw_change_deg")
            )
            direction_sign = _turn_direction_sign(intent.direction)
            observed_payload_progress_deg = max(
                0.0,
                direction_sign * observed_signed_payload_yaw_deg,
            )
            cumulative_signed_team_yaw_deg = (
                previous_signed_yaw_deg + observed_signed_team_yaw_deg
            )
            cumulative_signed_payload_yaw_deg = (
                previous_signed_payload_yaw_deg
                + observed_signed_payload_yaw_deg
            )
            cumulative_payload_progress_deg = (
                previous_payload_progress_deg + observed_payload_progress_deg
            )
            budget_reached = (
                cumulative_payload_progress_deg >= ROUTE_TURN_PROBE_YAW_DEG
            )
            passage_aligned = aligned
            alignment_required = budget_reached and not passage_aligned
            # The payload is the transported geometry. Robot-root yaw can lag
            # its long axis by tens of degrees during an outer/inner pivot, so
            # using root yaw here over-rotates the box before route selection
            # reopens.
            return {
                "schema_version": TURN_EPISODE_SCHEMA_VERSION,
                "active": not (passage_aligned or alignment_required),
                "status": (
                    PASSAGE_ALIGNED_STATUS
                    if passage_aligned
                    else PASSAGE_ALIGNMENT_REQUIRED_STATUS
                    if alignment_required
                    else "turning"
                ),
                "purpose": ROUTE_TURN_PURPOSE,
                "direction": intent.direction,
                "roles": dict(intent.roles),
                "primitive": intent.method,
                "last_angle_deg": intent.angle_deg,
                "accepted_turn_steps": previous_steps + 1,
                "last_observed_signed_team_yaw_deg": observed_signed_team_yaw_deg,
                "last_observed_signed_payload_yaw_deg": (
                    observed_signed_payload_yaw_deg
                ),
                "cumulative_signed_team_yaw_deg": cumulative_signed_team_yaw_deg,
                "cumulative_signed_payload_yaw_deg": (
                    cumulative_signed_payload_yaw_deg
                ),
                "cumulative_payload_turn_progress_deg": (
                    cumulative_payload_progress_deg
                ),
                "cumulative_turn_progress_deg": cumulative_payload_progress_deg,
                "cumulative_turn_progress_source": "physical_payload_yaw",
                "switch_allowed": alignment_required,
                "translation_pending": passage_aligned,
                "probe_direction": "",
                "probe_max_distance_m": MAX_ALIGNMENT_PROBE_DISTANCE_M,
                "latest_checkpoint_id": checkpoint_id,
                "completion_checkpoint_id": (
                    checkpoint_id if passage_aligned else ""
                ),
                "completion_trigger": (
                    alignment_source
                    if passage_aligned
                    else "turn_budget_reached_without_passage_alignment"
                    if alignment_required
                    else ""
                ),
                "latest_visual_alignment": (
                    "aligned_with_open_passage" if aligned else "not_aligned_or_uncertain"
                ),
                "latest_passage_posture": passage_posture,
                "passage_alignment_source": alignment_source,
            }
        if (
            not bool(current.get("active", False))
            and local_navigation.get("active", False)
            and local_navigation.get("reference")
            in {
                "final_target",
                "runtime_local_goal",
                "visible_connected_free_space",
            }
        ):
            if _local_navigation_direction_replan_required(report):
                return {
                    "schema_version": TURN_EPISODE_SCHEMA_VERSION,
                    "active": False,
                    "status": "local_direction_replan_required",
                    "purpose": "local_navigation_turn",
                    "reference": str(local_navigation.get("reference", "")),
                    "reference_geometry": str(
                        local_navigation.get("reference_geometry", "")
                    ),
                    "rejected_world_direction_xy": list(
                        local_navigation.get("goal_world_direction_xy", ())
                    ),
                    "rejected_goal_world_xy_m": list(
                        local_navigation.get("goal_world_xy_m", ())
                    ),
                    "rejected_direction": intent.direction,
                    "switch_allowed": True,
                    "implementation_switch_allowed": True,
                    "translation_pending": False,
                    "rejection_failure_type": report.failure_type,
                    "latest_checkpoint_id": checkpoint_id,
                    "rewrite_requirement": (
                        "re-observe the local navigation reference and declare a "
                        "new world direction before another turn"
                    ),
                }
            if report.failure_type in TURN_SCRIPT_REWRITE_FAILURES | {
                "alignment_regression",
                "no_payload_progress",
            }:
                return {
                    "schema_version": TURN_EPISODE_SCHEMA_VERSION,
                    "active": True,
                    "status": "local_execution_repair_required",
                    "purpose": "local_navigation_turn",
                    "reference": str(local_navigation.get("reference", "")),
                    "reference_geometry": str(
                        local_navigation.get("reference_geometry", "")
                    ),
                    "world_direction_xy": list(
                        local_navigation.get("goal_world_direction_xy", ())
                    ),
                    "goal_world_xy_m": list(
                        local_navigation.get("goal_world_xy_m", ())
                    ),
                    "direction": intent.direction,
                    "roles": dict(intent.roles),
                    "primitive": intent.method,
                    "last_angle_deg": intent.angle_deg,
                    "accepted_turn_steps": 0,
                    "switch_allowed": False,
                    "implementation_switch_allowed": True,
                    "translation_pending": False,
                    "script_failure_type": report.failure_type,
                    "failed_executable": {
                        "primitive": intent.method,
                        "direction": intent.direction,
                        "roles": dict(intent.roles),
                        "angle_deg": intent.angle_deg,
                    },
                    "signed_heading_error_before_deg": local_navigation.get(
                        "signed_heading_error_deg"
                    ),
                    "latest_checkpoint_id": checkpoint_id,
                    "rewrite_requirement": (
                        "preserve the declared local reference and signed turn; "
                        "repair the shared-heading implementation, temporary "
                        "roles, or one bounded magnitude"
                    ),
                }
        if bool(current.get("active", False)):
            if str(current.get("purpose", "")) == HEADING_CORRECTION_PURPOSE:
                updates: dict[str, Any] = {
                    "status": "correction_required",
                    "switch_allowed": False,
                    "rejection_failure_type": report.failure_type,
                    "latest_checkpoint_id": checkpoint_id,
                }
                if report.failure_type == "no_payload_progress":
                    previous_angle = _float(current.get("correction_angle_deg"))
                    if previous_angle > 0.0:
                        updates.update(
                            {
                                "status": "correction_retry_required",
                                "correction_angle_deg": min(
                                    5.0,
                                    max(
                                        previous_angle + 1.0,
                                        1.5 * previous_angle,
                                    ),
                                ),
                                "correction_retry_count": (
                                    int(current.get("correction_retry_count", 0))
                                    + 1
                                ),
                                "retry_reason": (
                                    "both robot roots must show material signed yaw; "
                                    "tune only the correction angle"
                                ),
                            }
                        )
                current.update(updates)
            elif (
                str(current.get("purpose", "")) == "local_navigation_turn"
                and _local_navigation_direction_replan_required(report)
            ):
                current.update(
                    {
                        "active": False,
                        "status": "local_direction_replan_required",
                        "switch_allowed": True,
                        "implementation_switch_allowed": True,
                        "translation_pending": False,
                        "rejected_direction": str(
                            current.get("direction", intent.direction)
                        ),
                        "rejected_world_direction_xy": list(
                            current.get("world_direction_xy", ())
                        ),
                        "rejection_failure_type": report.failure_type,
                        "latest_checkpoint_id": checkpoint_id,
                        "rewrite_requirement": (
                            "re-observe connected free space and declare a new local "
                            "world direction; do not preserve the rejected route sign"
                        ),
                    }
                )
            elif report.failure_type in TURN_SCRIPT_REWRITE_FAILURES or (
                str(current.get("purpose", "")) == "local_navigation_turn"
                and report.failure_type == "alignment_regression"
            ):
                local_navigation_turn = bool(
                    str(current.get("purpose", "")) == "local_navigation_turn"
                )
                current.update(
                    {
                        "status": (
                            "local_execution_repair_required"
                            if local_navigation_turn
                            else "script_rewrite_required"
                        ),
                        # Route sign and world direction stay locked.  Execution
                        # repair may still change the shared-heading
                        # implementation (static turn vs pivot), temporary
                        # roles, or one bounded magnitude.
                        "switch_allowed": not local_navigation_turn,
                        "implementation_switch_allowed": local_navigation_turn,
                        "script_failure_type": report.failure_type,
                        "failed_executable": {
                            "primitive": intent.method,
                            "direction": intent.direction,
                            "roles": dict(intent.roles),
                            "angle_deg": intent.angle_deg,
                        },
                        "rewrite_requirement": (
                            "preserve the locked local world direction and signed turn; "
                            "repair the executable by changing the shared-heading "
                            "implementation, temporary roles, or one bounded magnitude"
                            if local_navigation_turn
                            else (
                                "preserve the route goal but emit a materially different "
                                "turn transaction from current physics evidence"
                            )
                        ),
                        "latest_checkpoint_id": checkpoint_id,
                    }
                )
            elif report.failure_type in {"route_regression", "alignment_regression"}:
                current.update(
                    {
                        "status": "switch_allowed",
                        "switch_allowed": True,
                        "rejected_direction": str(
                            current.get("direction", intent.direction)
                        ),
                        "rejection_failure_type": report.failure_type,
                        "latest_checkpoint_id": checkpoint_id,
                    }
                )
        return current

    yaw_alignment_deltas = policy_yaw_alignment_deltas(policy)
    if yaw_alignment_deltas and bool(current.get("active", False)):
        if (
            report.success
            and str(current.get("purpose", "")) == ROUTE_TURN_PURPOSE
        ):
            current.update(
                {
                    "status": "probe_ready",
                    "switch_allowed": False,
                    "latest_checkpoint_id": checkpoint_id,
                    "heading_alignment_corrections": (
                        int(current.get("heading_alignment_corrections", 0)) + 1
                    ),
                    "last_heading_alignment_deltas_deg": yaw_alignment_deltas,
                    "probe_retry_count": 0,
                    "probe_retry_failure_type": "",
                }
            )
        return current

    probe_distance = policy_linear_probe_distance(policy)
    turn_clearance_setup = bool(
        probe_distance is not None
        and isinstance(policy.verification_contract, dict)
        and isinstance(
            policy.verification_contract.get("turn_clearance_replan"), dict
        )
        and policy.verification_contract["turn_clearance_replan"].get(
            "active", False
        )
    )
    if turn_clearance_setup and bool(current.get("active", False)):
        # A clearance setup never becomes a route-turn probe. Whether it succeeds
        # or rolls back, keep the original local target and signed turn alive.
        if report.success:
            current.update(
                {
                    "status": "turning",
                    "switch_allowed": False,
                    "translation_pending": False,
                    "latest_checkpoint_id": checkpoint_id,
                    "last_clearance_setup_checkpoint_id": checkpoint_id,
                    "clearance_setup_count": int(
                        current.get("clearance_setup_count", 0)
                    )
                    + 1,
                    "rewrite_requirement": (
                        "recompute authored-wall sweep from this checkpoint and resume "
                        "the preserved signed local-navigation turn"
                    ),
                }
            )
        else:
            current.update(
                {
                    "status": "turning",
                    "switch_allowed": False,
                    "translation_pending": False,
                    "latest_checkpoint_id": checkpoint_id,
                    "clearance_setup_failure_count": int(
                        current.get("clearance_setup_failure_count", 0)
                    )
                    + 1,
                    "clearance_setup_failure_type": report.failure_type,
                    "rewrite_requirement": (
                        "preserve the signed local-navigation turn and replan from "
                        "the restored checkpoint; do not reinterpret a failed "
                        "clearance translation as turn completion"
                    ),
                }
            )
        return current
    if (
        probe_distance is not None
        and bool(current.get("translation_pending", False))
    ):
        if report.success:
            completed = dict(current)
            completed.update(
                {
                    "active": False,
                    "status": "completed_with_translation",
                    "translation_pending": False,
                    "switch_allowed": False,
                    "translation_checkpoint_id": checkpoint_id,
                    "successful_translation_distance_m": probe_distance,
                }
            )
            return completed
        if str(current.get("purpose", "")) == "local_navigation_turn":
            retry = dict(current)
            retry.update(
                {
                    "active": False,
                    "status": "local_heading_aligned",
                    "translation_pending": True,
                    "switch_allowed": False,
                    "probe_retry_count": int(
                        current.get("probe_retry_count", 0)
                    )
                    + 1,
                    "probe_retry_failure_type": report.failure_type,
                    "latest_checkpoint_id": checkpoint_id,
                    "rewrite_requirement": (
                        "preserve the locked local world direction and repair "
                        "the rejected translation without starting another turn"
                    ),
                }
            )
            return retry
        reopened = dict(current)
        alignment_verified = _episode_has_verified_passage_alignment(current)
        reopened.update(
            {
                "active": False,
                "status": (
                    PASSAGE_ALIGNED_STATUS
                    if alignment_verified
                    else PASSAGE_ALIGNMENT_REQUIRED_STATUS
                ),
                "translation_pending": alignment_verified,
                "switch_allowed": True,
                "probe_retry_count": int(current.get("probe_retry_count", 0)) + 1,
                "probe_retry_failure_type": report.failure_type,
                "latest_checkpoint_id": checkpoint_id,
                "rewrite_requirement": (
                    "reselect a materially different executable transaction from "
                    "the restored checkpoint; do not replay the rejected linear AST"
                ),
            }
        )
        return reopened
    if probe_distance is not None and bool(current.get("active", False)):
        if report.success and str(current.get("status", "")) == "probe_ready":
            completed = dict(current)
            completed.update(
                {
                    "active": False,
                    "status": "completed",
                    "switch_allowed": False,
                    "completion_checkpoint_id": checkpoint_id,
                    "successful_probe_distance_m": probe_distance,
                }
            )
            return completed
        if report.failure_type in {"route_regression", "alignment_regression"}:
            current.update(
                {
                    "status": "switch_allowed",
                    "switch_allowed": True,
                    "rejection_failure_type": report.failure_type,
                    "latest_checkpoint_id": checkpoint_id,
                }
            )
        else:
            # A coordination or execution failure does not prove that the
            # selected route direction is wrong. Retry once without adding a
            # turn, then reopen strategy selection instead of replaying the
            # same physical transaction indefinitely.
            retry_count = int(current.get("probe_retry_count", 0)) + 1
            retry_exhausted = retry_count >= 2
            current.update(
                {
                    "status": (
                        "switch_allowed"
                        if retry_exhausted
                        else "probe_ready"
                    ),
                    "switch_allowed": retry_exhausted,
                    "probe_retry_count": retry_count,
                    "probe_retry_failure_type": report.failure_type,
                    "latest_checkpoint_id": checkpoint_id,
                }
            )
    return current


def _local_navigation_direction_replan_required(
    report: EvaluationReport,
) -> bool:
    """Separate a bad route declaration from a bad physical turn execution."""

    if report.failure_type in {"route_regression", "goal_regression"}:
        return True
    if report.failure_type != "alignment_regression":
        return False
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    transition = evidence.get("local_navigation_transition", {})
    transition = transition if isinstance(transition, dict) else {}
    return bool(
        transition.get("available", False)
        and not transition.get("requested_turn_matches_local_error", True)
    )


def _straight_heading_correction_episode(
    report: EvaluationReport,
    *,
    checkpoint_id: str,
) -> dict[str, Any] | None:
    if report.failure_type != "straightness_regression":
        return None
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    straight = evidence.get("straight_transport", {})
    straight = straight if isinstance(straight, dict) else {}
    if str(straight.get("correction_method", "")) != COORDINATED_PIVOT:
        return None
    direction = str(straight.get("correction_turn_direction", "")).strip().lower()
    if direction not in TURN_DIRECTIONS:
        return None
    angle_deg = _float(straight.get("correction_turn_angle_deg"))
    yaw_deltas = straight.get("correction_yaw_deltas_deg", {})
    yaw_deltas = yaw_deltas if isinstance(yaw_deltas, dict) else {}
    if angle_deg <= 0.0 or set(yaw_deltas) != {"front", "back"}:
        return None
    magnitudes = {
        robot_id: abs(_float(yaw_deltas.get(robot_id)))
        for robot_id in ("front", "back")
    }
    if magnitudes["front"] == magnitudes["back"]:
        return None
    outer = max(magnitudes, key=magnitudes.get)
    inner = "back" if outer == "front" else "front"
    return {
        "schema_version": 1,
        "active": True,
        "status": "correction_required",
        "purpose": HEADING_CORRECTION_PURPOSE,
        "direction": direction,
        "roles": {inner: "inner", outer: "outer"},
        "primitive": COORDINATED_PIVOT,
        "correction_angle_deg": angle_deg,
        "accepted_turn_steps": 0,
        "switch_allowed": False,
        "probe_direction": "forward",
        "probe_max_distance_m": MAX_ALIGNMENT_PROBE_DISTANCE_M,
        "latest_checkpoint_id": checkpoint_id,
        "source_failure_type": report.failure_type,
    }


def with_turn_episode(
    runtime_state: dict[str, Any],
    episode: dict[str, Any] | None,
) -> dict[str, Any]:
    enriched = dict(runtime_state)
    enriched["turn_episode"] = dict(
        episode if isinstance(episode, dict) else empty_turn_episode()
    )
    return enriched


def observe_passage_alignment(
    episode: dict[str, Any] | None,
    *,
    visual_report: VisualVerifierReport | None,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Refresh a route turn from the current trusted global view."""

    current = normalize_turn_episode(episode)
    if str(current.get("purpose", "")) != ROUTE_TURN_PURPOSE:
        return current

    status = str(current.get("status", ""))
    active_turn = bool(current.get("active", False)) and status == "turning"
    active_probe_ready = (
        bool(current.get("active", False)) and status == "probe_ready"
    )
    refreshable_completed_turn = (
        not bool(current.get("active", False))
        and str(current.get("status", ""))
        in {
            "completed",
            "probe_ready",
            PASSAGE_ALIGNMENT_REQUIRED_STATUS,
            PASSAGE_ALIGNED_STATUS,
        }
    )
    if not (active_turn or active_probe_ready or refreshable_completed_turn):
        return current

    visual_aligned = _visually_aligned_with_open_passage(visual_report)
    visual_risk_confirmed = bool(
        visual_report is not None
        and isinstance(visual_report.evidence, dict)
        and visual_report.evidence.get("route_direction_risk_confirmed", False)
    )
    trusted_alignment_preserved = bool(
        not visual_risk_confirmed
        and str(current.get("status", "")) == PASSAGE_ALIGNED_STATUS
        and bool(current.get("translation_pending", False))
        and _episode_has_verified_passage_alignment(current)
    )
    aligned = visual_aligned or trusted_alignment_preserved
    posture = (
        _visual_passage_posture(visual_report)
        if visual_aligned
        else str(current.get("latest_passage_posture", "")).strip()
        if trusted_alignment_preserved
        else _visual_passage_posture(visual_report)
    )
    alignment_source = (
        "global_visual_passage_alignment"
        if visual_aligned
        else str(current.get("passage_alignment_source", "")).strip()
        if trusted_alignment_preserved
        else ""
    )
    if active_turn and not aligned:
        current.update(
            {
                "latest_visual_alignment": "not_aligned_or_uncertain",
                "latest_passage_posture": posture,
            }
        )
        return current

    current.update(
        {
            "active": False,
            "status": (
                PASSAGE_ALIGNED_STATUS
                if aligned
                else PASSAGE_ALIGNMENT_REQUIRED_STATUS
            ),
            "switch_allowed": not aligned,
            "translation_pending": aligned,
            "latest_checkpoint_id": checkpoint_id,
            "completion_checkpoint_id": checkpoint_id if aligned else "",
            "latest_visual_alignment": (
                "aligned_with_open_passage"
                if aligned
                else "not_aligned_or_uncertain"
            ),
            "latest_passage_posture": posture,
            "completion_trigger": (
                "global_visual_passage_alignment_before_yaw_budget"
                if visual_aligned and active_turn
                else "global_visual_passage_alignment_after_heading_correction"
                if visual_aligned and active_probe_ready
                else "global_visual_passage_alignment"
                if visual_aligned
                else str(current.get("completion_trigger", "")).strip()
                if trusted_alignment_preserved
                else "current_global_view_requires_passage_alignment"
            ),
            "passage_alignment_source": alignment_source,
        }
    )
    return current


def _visually_aligned_with_open_passage(
    visual_report: VisualVerifierReport | None,
) -> bool:
    if visual_report is None or not isinstance(visual_report.evidence, dict):
        return False
    posture = _visual_passage_posture(visual_report)
    return (
        (
            posture in ALIGNED_PASSAGE_POSTURES
            or (
                not posture
                and str(
                    visual_report.evidence.get(
                        "team_heading_relative_to_visible_free_space",
                        "",
                    )
                )
                == "aligned_with_open_passage"
            )
        )
        and not bool(
            visual_report.evidence.get("route_direction_risk_confirmed", False)
        )
    )


def _physics_aligned_passage_posture(
    report: EvaluationReport,
    *,
    visual_report: VisualVerifierReport | None,
) -> str:
    """Classify planar passage posture without confusing payload roll with yaw."""

    if not report.success:
        return ""
    evidence = report.evidence if isinstance(report.evidence, dict) else {}
    if bool(evidence.get("route_discovery", False)):
        # Active-stage XY goals are hidden supervision during route discovery.
        # Only the fixed global view may close a discovered passage turn.
        return ""
    if (
        visual_report is not None
        and isinstance(visual_report.evidence, dict)
        and bool(
            visual_report.evidence.get(
                "route_direction_risk_confirmed", False
            )
        )
    ):
        return ""
    metrics = report.metrics if isinstance(report.metrics, dict) else {}
    required = {
        "robot_pair_axis_alignment_active_goal",
        "payload_long_axis_alignment_active_goal",
        "team_forward_alignment_active_goal",
        "robot_heading_disagreement_after_deg",
    }
    if not required.issubset(metrics):
        return ""
    pair_alignment = abs(
        _float(metrics.get("robot_pair_axis_alignment_active_goal"))
    )
    payload_alignment = abs(
        _float(metrics.get("payload_long_axis_alignment_active_goal"))
    )
    heading_split_deg = abs(
        _float(metrics.get("robot_heading_disagreement_after_deg"))
    )
    if (
        min(pair_alignment, payload_alignment)
        < PHYSICS_PASSAGE_AXIS_ALIGNMENT_MIN
        or heading_split_deg > PHYSICS_ROBOT_HEADING_SPLIT_MAX_DEG
    ):
        return ""
    team_forward_alignment = abs(
        _float(metrics.get("team_forward_alignment_active_goal"))
    )
    return (
        "longitudinal_aligned"
        if team_forward_alignment >= LONGITUDINAL_ALIGNMENT_COSINE_MIN
        else "lateral_aligned"
    )


def _visual_passage_posture(
    visual_report: VisualVerifierReport | None,
) -> str:
    if visual_report is None or not isinstance(visual_report.evidence, dict):
        return ""
    return str(
        visual_report.evidence.get(
            "team_payload_posture_relative_to_visible_passage",
            "",
        )
    ).strip()


def _episode_has_verified_passage_alignment(episode: dict[str, Any]) -> bool:
    posture = str(episode.get("latest_passage_posture", "")).strip()
    if posture:
        return posture in ALIGNED_PASSAGE_POSTURES
    return (
        str(episode.get("latest_visual_alignment", "")).strip()
        == "aligned_with_open_passage"
    )


def _policy_calls(policy: PolicyScript) -> dict[str, tuple[str, dict[str, Any]]]:
    scripts = policy.robot_scripts
    if not scripts:
        return {}
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for script in scripts:
        call = _single_task_call(script.code)
        if call is None:
            return {}
        calls[script.robot_id] = call
    return calls


def _single_task_call(source: str) -> tuple[str, dict[str, Any]] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    calls: list[tuple[str, dict[str, Any]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if not isinstance(owner, ast.Name) or owner.id != "api":
            continue
        if node.func.attr == "stop_all":
            continue
        try:
            arguments = {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
        except (TypeError, ValueError):
            return None
        calls.append((node.func.attr, arguments))
    return calls[0] if len(calls) == 1 else None


def _episode_roles(episode: dict[str, Any]) -> dict[str, str]:
    roles = episode.get("roles", {})
    return {
        str(robot_id): str(role)
        for robot_id, role in roles.items()
    } if isinstance(roles, dict) else {}


def _selected_roles(
    method: str,
    robot_parameters: dict[str, Any],
) -> dict[str, str]:
    if method in STATIC_TURN_METHODS:
        return {}
    if method == COORDINATED_PIVOT:
        return {
            robot_id: str(parameters.get("role", ""))
            for robot_id, parameters in robot_parameters.items()
            if isinstance(parameters, dict)
        }
    radii = {
        robot_id: _float(parameters.get("radius_m"))
        for robot_id, parameters in robot_parameters.items()
        if isinstance(parameters, dict)
    }
    if set(radii) != {"front", "back"} or radii["front"] == radii["back"]:
        return {}
    return {
        "front": "outer" if radii["front"] > radii["back"] else "inner",
        "back": "outer" if radii["back"] > radii["front"] else "inner",
    }


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _turn_direction_sign(direction: str) -> float:
    if direction == "counterclockwise":
        return 1.0
    if direction == "clockwise":
        return -1.0
    return 0.0


__all__ = [
    "ALIGNED_PASSAGE_POSTURES",
    "HOST_WALL_CLEARANCE_ESCAPE_MARKER",
    "HOST_WALL_CLEARANCE_ESCAPE_PARAMETER_SOURCE",
    "HOST_WALL_CLEARANCE_ESCAPE_SELECTION_SOURCE",
    "HOST_WALL_TURN_ESCAPE_MARKER",
    "HOST_WALL_TURN_ESCAPE_PARAMETER_SOURCE",
    "HOST_WALL_TURN_ESCAPE_SELECTION_SOURCE",
    "MAX_ALIGNMENT_PROBE_DISTANCE_M",
    "PASSAGE_ALIGNED_STATUS",
    "PASSAGE_ALIGNMENT_REQUIRED_STATUS",
    "ROUTE_TURN_PROBE_YAW_DEG",
    "ROLE_BASED_TURN_METHODS",
    "STATIC_TURN_DIRECTIONS",
    "STATIC_TURN_METHODS",
    "TURN_METHODS",
    "TurnEpisodeError",
    "empty_turn_episode",
    "normalize_turn_episode",
    "observe_passage_alignment",
    "policy_linear_probe_direction",
    "policy_linear_probe_distance",
    "policy_turn_intent",
    "update_turn_episode",
    "validate_policy_for_turn_episode",
    "validate_selected_parameters_for_turn_episode",
    "with_turn_episode",
]
