"""Single-source mutation masks for prompts, schemas, and validators."""

from __future__ import annotations

import json
from typing import Any

from .coordination_specs import (
    coordination_modes_for_method,
    coordination_primitive_method_spec,
    coordination_relationship_can_change,
    COORDINATED_PIVOT,
)


MUTATION_LEVEL_ORDER = (
    "strategy_mutation",
    "coordination_mutation",
    "parameter_tuning",
)

# Recovery widens monotonically from literal repair to coordination structure
# and finally to a new strategy family.  ``MUTATION_LEVEL_ORDER`` remains the
# prompt/display order; rank is the recovery partial order.
MUTATION_LEVEL_RANK = {
    "parameter_tuning": 0,
    "coordination_mutation": 1,
    "strategy_mutation": 2,
}
MUTATION_LEVEL_BY_RANK = {
    rank: level for level, rank in MUTATION_LEVEL_RANK.items()
}

_PARAMETER_FIELDS_BY_METHOD = {
    "coordinated_linear_transport": ("distance_m",),
    "coordinated_arc_transport": ("angle_deg",),
    "coordinated_pivot": ("angle_deg",),
    "coordinated_alignment": ("delta",),
    "carry_forward": ("distance_m",),
    "carry_backward": ("distance_m",),
    "carry_lateral_left": ("distance_m",),
    "carry_lateral_right": ("distance_m",),
    "turn_clockwise": ("yaw_deg",),
    "turn_counterclockwise": ("yaw_deg",),
    "place_crouch": ("place_height",),
}

_LINEAR_OBJECTIVE_BY_METHOD = {
    "carry_forward": "forward",
    "carry_backward": "backward",
    "carry_lateral_left": "left",
    "carry_lateral_right": "right",
}

_EXECUTABLE_STRATEGY_FIELDS = (
    "primitive",
    "semantic_objective",
    "coordination_mode",
)
_PROSE_ONLY_STRATEGY_FIELDS = (
    "formation_strategy",
    "synchronization_intent",
)


def _normalized_linear_objective(strategy: dict[str, Any]) -> str:
    method = str(strategy.get("primitive", "")).strip()
    if method in _LINEAR_OBJECTIVE_BY_METHOD:
        return _LINEAR_OBJECTIVE_BY_METHOD[method]
    if method == "coordinated_linear_transport":
        objective = str(strategy.get("semantic_objective", "")).strip()
        if objective in {"forward", "backward", "left", "right"}:
            return objective
    return ""


def host_effective_mutation_delta(
    candidate: dict[str, Any],
    *,
    parent_strategy: dict[str, Any],
    parent_effective_parameters: dict[str, Any] | None = None,
    child_effective_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the executable child-parent delta without trusting model claims."""

    parent = parent_strategy if isinstance(parent_strategy, dict) else {}
    child = candidate.get("child_strategy", {})
    child = child if isinstance(child, dict) else {}
    changed_strategy_fields = [
        field
        for field in _EXECUTABLE_STRATEGY_FIELDS
        if str(parent.get(field, "")).strip()
        != str(child.get(field, "")).strip()
    ]
    parent_roles = _roles(parent)
    child_roles = _roles(child)
    roles_changed = parent_roles != child_roles
    prose_only_fields = [
        field
        for field in _PROSE_ONLY_STRATEGY_FIELDS
        if str(parent.get(field, "")).strip()
        != str(child.get(field, "")).strip()
    ]
    before_parameters = (
        parent_effective_parameters
        if isinstance(parent_effective_parameters, dict)
        else {}
    )
    after_parameters = (
        child_effective_parameters
        if isinstance(child_effective_parameters, dict)
        else {}
    )
    changed_parameter_fields = sorted(
        {
            str(field)
            for field in set(before_parameters) | set(after_parameters)
            if before_parameters.get(field) != after_parameters.get(field)
        }
    )
    level = str(candidate.get("mutation_level", "")).strip()
    parameter_validation = (
        "complete"
        if parent_effective_parameters is not None
        and child_effective_parameters is not None
        else "pending_coder"
        if level == "parameter_tuning"
        else "not_applicable"
    )
    effective = bool(
        changed_strategy_fields or roles_changed or changed_parameter_fields
    )
    if level == "strategy_mutation":
        effective = bool(
            set(changed_strategy_fields) & {"primitive", "semantic_objective"}
        )
    elif level == "coordination_mutation":
        effective = bool("coordination_mode" in changed_strategy_fields or roles_changed)
    elif level == "parameter_tuning" and parameter_validation != "complete":
        effective = False
    elif level == "parameter_tuning":
        effective = bool(changed_parameter_fields)
    return {
        "source": "host_executable_projection_diff",
        "mutation_level": level,
        "changed_strategy_fields": changed_strategy_fields,
        "roles_changed": roles_changed,
        "parent_roles": parent_roles,
        "child_roles": child_roles,
        "changed_parameter_fields": changed_parameter_fields,
        "parameter_validation": parameter_validation,
        "prose_only_fields": prose_only_fields,
        "effective": effective,
    }


def no_effective_mutation_feedback(
    *,
    mutation_level: str,
    delta: dict[str, Any],
) -> dict[str, Any]:
    """Return the machine-readable correction contract for a prose-only child."""

    return {
        "code": "NO_EFFECTIVE_MUTATION",
        "required_level": str(mutation_level),
        "unchanged_executable_fields": [
            "primitive",
            "semantic_objective",
            "coordination_mode",
            "roles",
            "effective_parameters",
        ],
        "observed_host_delta": delta,
        "prose_only_change": bool(delta.get("prose_only_fields")),
        "note": (
            "Changing rationale, formation_strategy text, synchronization_intent, "
            "IDs, or phase labels does not constitute an executable mutation."
        ),
    }


def _no_effective_mutation_error(level: str, delta: dict[str, Any]) -> ValueError:
    feedback = no_effective_mutation_feedback(
        mutation_level=level,
        delta=delta,
    )
    return ValueError(
        "NO_EFFECTIVE_MUTATION "
        + json.dumps(feedback, ensure_ascii=True, sort_keys=True)
    )


def build_mutation_contract(parent_strategy: dict[str, Any]) -> dict[str, Any]:
    """Build the only mutable mask used for one rejected parent strategy."""

    parent = parent_strategy if isinstance(parent_strategy, dict) else {}
    method = str(parent.get("primitive", "")).strip()
    mode = str(parent.get("coordination_mode", "")).strip()
    parameter_fields = list(_PARAMETER_FIELDS_BY_METHOD.get(method, ()))
    coordination_available = bool(
        method and mode and coordination_relationship_can_change(method, mode)
    )
    levels = {
        "parameter_tuning": {
            "available": bool(parameter_fields),
            "mutable": ["exactly one action parameter"],
            "allowed_parameter_fields": parameter_fields,
            "immutable": [
                "primitive",
                "semantic_objective",
                "formation_strategy",
                "coordination_mode",
                "temporary_front_back_roles",
            ],
        },
        "coordination_mutation": {
            "available": coordination_available,
            "mutable": ["coordination_mode", "temporary_front_back_roles"],
            "allowed_parameter_fields": [],
            "immutable": [
                "primitive",
                "semantic_objective",
                "formation_strategy",
                "action_parameters",
            ],
            "unavailable_reason": (
                "the parent primitive exposes no alternative executable mode or "
                "temporary-role relation"
                if not coordination_available
                else ""
            ),
        },
        "strategy_mutation": {
            "available": True,
            "mutable": [
                "primitive",
                "semantic_objective_or_direction",
                "formation_strategy",
                "dependent_coordination_mode_and_roles",
                "parameters_required_by_the_new_strategy",
            ],
            "allowed_parameter_fields": [],
            "immutable": [],
        },
    }
    available_levels = [
        level for level in MUTATION_LEVEL_ORDER if levels[level]["available"]
    ]
    return {
        "schema_version": 1,
        "parent_primitive": method,
        "parent_semantic_objective": str(
            parent.get("semantic_objective", "")
        ).strip(),
        "parent_coordination_mode": mode,
        "available_levels": available_levels,
        "levels": levels,
        "always_preserve": [
            "global mission",
            "current phase goal",
            "accepted checkpoint progress",
            "payload lifecycle invariants",
            "physical safety",
            "immutable robot IDs front and back",
        ],
    }


def apply_minimum_mutation_rank(
    contract: dict[str, Any],
    minimum_rank: int,
) -> dict[str, Any]:
    """Remove narrower recovery levels after a complete semantic failure.

    This changes only the mutation scope exposed to Evolution Manager.  It
    never selects a primitive, direction, coordination mode, role, or literal.
    """

    payload = contract if isinstance(contract, dict) else {}
    minimum_rank = max(0, min(max(MUTATION_LEVEL_BY_RANK), int(minimum_rank)))
    levels = payload.get("levels", {})
    levels = levels if isinstance(levels, dict) else {}
    for level, rank in MUTATION_LEVEL_RANK.items():
        if rank >= minimum_rank:
            continue
        details = levels.get(level, {})
        details = dict(details) if isinstance(details, dict) else {}
        details["available"] = False
        details["below_minimum_mutation_rank"] = True
        details["unavailable_reason"] = (
            "checkpoint-local recovery already failed at this or a broader "
            "semantic scope; mutation scope must expand monotonically"
        )
        levels[level] = details
    payload["levels"] = levels
    payload["minimum_mutation_rank"] = minimum_rank
    payload["minimum_mutation_level"] = MUTATION_LEVEL_BY_RANK[minimum_rank]
    payload["available_levels"] = [
        level
        for level in payload.get("available_levels", ())
        if MUTATION_LEVEL_RANK.get(str(level).strip(), -1) >= minimum_rank
    ]
    payload["allowed_mutation_levels"] = list(payload["available_levels"])
    return payload


def validate_manager_level(
    candidate: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Require Evolution Manager to select one level available in the mask."""

    level = str(candidate.get("mutation_level", "")).strip()
    available = {
        str(item).strip()
        for item in contract.get("available_levels", ())
        if str(item).strip()
    }
    if level not in available:
        details = contract.get("levels", {}).get(level, {})
        details = details if isinstance(details, dict) else {}
        if details.get("infeasible_at_checkpoint"):
            raise ValueError(
                f"The selected {level} mutation has no contract-valid effective "
                "child at the current checkpoint. Reselect a mutation level "
                "based on the current evidence."
            )
        reason = str(details.get("unavailable_reason", "")).strip()
        suffix = f": {reason}" if reason else ""
        raise ValueError(
            f"mutation_contract marks {level!r} unavailable for the rejected "
            f"parent; available_levels={sorted(available)}{suffix}"
        )


def validate_mutation_transition(
    candidate: dict[str, Any],
    *,
    parent_strategy: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Enforce the selected level using the same mutable mask shown to the model."""

    validate_manager_level(candidate, contract)
    level = str(candidate.get("mutation_level", "")).strip()
    child = candidate.get("child_strategy", {})
    child = child if isinstance(child, dict) else {}
    parent = parent_strategy if isinstance(parent_strategy, dict) else {}
    # Mutable-mask enforcement is defined over the executable projection.
    # Prose is retained in ``host_effective_delta`` for audit, but a wording
    # change must never invalidate an otherwise legal executable mutation.
    fields = _EXECUTABLE_STRATEGY_FIELDS
    changed = {
        field
        for field in fields
        if str(parent.get(field, "")).strip() != str(child.get(field, "")).strip()
    }
    parent_roles = _roles(parent)
    child_roles = _roles(child)
    roles_changed = parent_roles != child_roles
    delta = host_effective_mutation_delta(
        candidate,
        parent_strategy=parent,
    )
    candidate["host_effective_delta"] = delta

    if level == "parameter_tuning":
        if changed or roles_changed:
            raise ValueError(
                "parameter_tuning mutable mask permits action parameters only; "
                f"changed strategy fields={sorted(changed)} roles_changed={roles_changed}"
            )
        dimensions = {
            str(item).strip()
            for item in candidate.get("changed_dimensions", ())
            if str(item).strip()
        }
        allowed = set(
            contract.get("levels", {})
            .get(level, {})
            .get("allowed_parameter_fields", ())
        )
        if dimensions and (len(dimensions) != 1 or not dimensions <= allowed):
            raise ValueError(
                "parameter_tuning must name exactly one allowed parameter field; "
                f"allowed={sorted(allowed)}, got={sorted(dimensions)}"
            )
        if not dimensions:
            raise _no_effective_mutation_error(level, delta)
        return

    if level == "coordination_mutation":
        forbidden_changes = changed - {"coordination_mode"}
        if forbidden_changes:
            raise ValueError(
                "coordination_mutation mutable mask preserves primitive, semantic "
                "objective; changed executable fields="
                f"{sorted(forbidden_changes)}"
            )
        if "coordination_mode" not in changed and not roles_changed:
            raise _no_effective_mutation_error(level, delta)
        return

    strategy_changed = bool(changed & {"primitive", "semantic_objective"})
    if not strategy_changed:
        raise _no_effective_mutation_error(level, delta)
    parent_linear_objective = _normalized_linear_objective(parent)
    child_linear_objective = _normalized_linear_objective(child)
    if (
        parent_linear_objective
        and child_linear_objective == parent_linear_objective
        and str(parent.get("primitive", "")).strip()
        != str(child.get("primitive", "")).strip()
    ):
        raise ValueError(
            "strategy_mutation cannot rename the same body-frame linear objective "
            f"({parent_linear_objective}) through a different SDK primitive; choose "
            "a genuinely different motion objective or coordination primitive"
        )


MUTATION_SCOPE_FEASIBLE = "FEASIBLE"
MUTATION_SCOPE_INFEASIBLE = "INFEASIBLE"
MUTATION_SCOPE_UNKNOWN = "UNKNOWN"

_SELECTED_SCOPE_INFEASIBLE_REASON = (
    "The selected mutation-level operator has no contract-valid effective child "
    "at the current checkpoint. Reselect a mutation level based on the current "
    "evidence."
)


def check_selected_mutation_scope(
    selected_level: str,
    *,
    parent_strategy: dict[str, Any],
    host_pivot_role_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove whether the selected mutation operator's discrete child set is empty.

    FEASIBLE means at least one contract-valid effective delta exists.
    INFEASIBLE is only returned when every editable field of that operator is a
    finite domain that is already exhausted or illegal. UNKNOWN means emptiness
    cannot be proven without enumerating maneuvers or continuous parameters.
    This does not rank, recommend, or list replacement actions.
    """

    level = str(selected_level).strip()
    parent = parent_strategy if isinstance(parent_strategy, dict) else {}
    method = str(parent.get("primitive", "")).strip()
    mode = str(parent.get("coordination_mode", "")).strip()
    objective = str(parent.get("semantic_objective", "")).strip()
    if level == "parameter_tuning":
        fields = _PARAMETER_FIELDS_BY_METHOD.get(method, ())
        if not fields:
            return _scope_status(
                MUTATION_SCOPE_INFEASIBLE,
                level,
                "this primitive exposes no parameter-tuning field",
            )
        return _scope_status(
            MUTATION_SCOPE_UNKNOWN,
            level,
            "parameter values are not a fully enumerable host contract",
        )
    if level == "strategy_mutation":
        return _scope_status(
            MUTATION_SCOPE_UNKNOWN,
            level,
            "strategy children are not proven empty without enumerating maneuvers",
        )
    if level != "coordination_mutation":
        return _scope_status(
            MUTATION_SCOPE_UNKNOWN,
            level,
            "unrecognized mutation level",
        )
    if not method or not mode:
        return _scope_status(
            MUTATION_SCOPE_UNKNOWN,
            level,
            "parent coordination fields are incomplete",
        )
    alternative_modes = {
        item for item in coordination_modes_for_method(method) if item != mode
    }
    if alternative_modes:
        return _scope_status(
            MUTATION_SCOPE_FEASIBLE,
            level,
            "an alternative executable coordination mode exists",
        )
    role_reassignment = False
    try:
        role_reassignment = bool(
            coordination_primitive_method_spec(method).get(
                "supports_role_reassignment",
                False,
            )
        )
    except (KeyError, TypeError, ValueError):
        role_reassignment = False
    if not role_reassignment:
        return _scope_status(
            MUTATION_SCOPE_INFEASIBLE,
            level,
            _SELECTED_SCOPE_INFEASIBLE_REASON,
        )
    parent_roles = _roles(parent)
    contract = (
        host_pivot_role_contract
        if isinstance(host_pivot_role_contract, dict)
        else {}
    )
    assignments = contract.get("role_assignment_by_direction", {})
    assignments = assignments if isinstance(assignments, dict) else {}
    fixed_roles = assignments.get(objective, {})
    fixed_roles = (
        {
            robot_id: str(role).strip()
            for robot_id, role in fixed_roles.items()
        }
        if isinstance(fixed_roles, dict)
        else {}
    )
    if (
        method != COORDINATED_PIVOT
        or set(fixed_roles) != {"front", "back"}
        or set(fixed_roles.values()) != {"inner", "outer"}
    ):
        return _scope_status(
            MUTATION_SCOPE_UNKNOWN,
            level,
            "temporary-role alternatives are not a proven-empty finite contract",
        )
    if parent_roles != fixed_roles:
        return _scope_status(
            MUTATION_SCOPE_FEASIBLE,
            level,
            "a contract-valid role assignment differs from the parent",
        )
    return _scope_status(
        MUTATION_SCOPE_INFEASIBLE,
        level,
        _SELECTED_SCOPE_INFEASIBLE_REASON,
    )


def apply_infeasible_mutation_level(
    contract: dict[str, Any],
    level: str,
    *,
    reason: str = _SELECTED_SCOPE_INFEASIBLE_REASON,
) -> dict[str, Any]:
    """Mark one previously selected mutation operator unavailable at this checkpoint."""

    payload = contract if isinstance(contract, dict) else {}
    levels = payload.get("levels", {})
    levels = levels if isinstance(levels, dict) else {}
    details = levels.get(level, {})
    details = dict(details) if isinstance(details, dict) else {}
    details["available"] = False
    details["infeasible_at_checkpoint"] = True
    details["unavailable_reason"] = str(reason).strip() or _SELECTED_SCOPE_INFEASIBLE_REASON
    levels[level] = details
    payload["levels"] = levels
    payload["available_levels"] = [
        item
        for item in payload.get("available_levels", ())
        if str(item).strip() != str(level).strip()
    ]
    return payload


def _scope_status(status: str, level: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "selected_level": str(level).strip(),
        "reason": str(reason).strip(),
        "enumerates_actions": False,
    }


def _roles(strategy: dict[str, Any]) -> dict[str, str]:
    assignment = strategy.get("team_assignment", {})
    assignment = assignment if isinstance(assignment, dict) else {}
    return {
        robot_id: str(
            assignment.get(robot_id, {}).get("role", "")
            if isinstance(assignment.get(robot_id), dict)
            else ""
        ).strip()
        for robot_id in ("front", "back")
    }
