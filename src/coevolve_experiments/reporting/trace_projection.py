"""Canonical executable projections used by Figure-2 lineage validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


class PolicyProjectionError(ValueError):
    """Raised when a policy lacks the executable fields needed for lineage."""


_COMMITTED_FIELDS = (
    "phase_id",
    "subtask_id",
    "primitive",
    "semantic_objective",
    "coordination_mode",
)
_ASSIGNMENT_FIELDS = (
    "role",
    "coordination_mode",
    "synchronization_group",
    "phase_id",
    "contact_intent",
)
_SCRIPT_FIELDS = (
    "coordination_mode",
    "synchronization_group",
    "phase_id",
    "contact_intent",
)


def executable_policy_projection(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the host-checkable executable surface of one team policy.

    Rationale and free-form objectives are deliberately excluded.  Code bytes,
    the committed primitive/semantics, and concrete robot-role coordination
    fields are included, so a prose-only rewrite has an identical projection.
    """

    verification = _mapping(policy.get("verification_contract"), "verification_contract")
    committed = _mapping(verification.get("committed_subtask"), "committed_subtask")
    committed_projection = {
        field: _required_text(committed.get(field), f"committed_subtask.{field}")
        for field in _COMMITTED_FIELDS
    }

    assignments_payload = policy.get("assignments")
    if not isinstance(assignments_payload, list) or not assignments_payload:
        raise PolicyProjectionError("policy.assignments must be a non-empty list")
    assignments: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(assignments_payload):
        item = _mapping(raw, f"assignments[{index}]")
        robot_id = _required_text(item.get("robot_id"), f"assignments[{index}].robot_id")
        if robot_id in assignments:
            raise PolicyProjectionError(f"duplicate assignment robot_id={robot_id!r}")
        assignments[robot_id] = {
            field: _required_text(item.get(field), f"assignments[{index}].{field}")
            for field in _ASSIGNMENT_FIELDS
        }

    scripts_payload = policy.get("robot_scripts")
    if not isinstance(scripts_payload, list) or not scripts_payload:
        raise PolicyProjectionError("policy.robot_scripts must be a non-empty list")
    scripts: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(scripts_payload):
        item = _mapping(raw, f"robot_scripts[{index}]")
        robot_id = _required_text(item.get("robot_id"), f"robot_scripts[{index}].robot_id")
        if robot_id in scripts:
            raise PolicyProjectionError(f"duplicate robot script robot_id={robot_id!r}")
        code = _required_text(item.get("code"), f"robot_scripts[{index}].code")
        scripts[robot_id] = {
            **{
                field: _required_text(
                    item.get(field),
                    f"robot_scripts[{index}].{field}",
                )
                for field in _SCRIPT_FIELDS
            },
            "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        }
    if set(assignments) != set(scripts):
        raise PolicyProjectionError(
            "assignment and robot-script robot IDs must match exactly"
        )

    team_code = _required_text(policy.get("code"), "policy.code")
    return {
        "committed_subtask": committed_projection,
        "assignments": {key: assignments[key] for key in sorted(assignments)},
        "robot_scripts": {key: scripts[key] for key in sorted(scripts)},
        "team_code_sha256": hashlib.sha256(team_code.encode("utf-8")).hexdigest(),
    }


def projection_sha256(projection: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def changed_projection_dimensions(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return deterministic leaf paths changed between two projections."""

    parent_leaves = _flatten(parent)
    child_leaves = _flatten(child)
    return tuple(
        key
        for key in sorted(set(parent_leaves) | set(child_leaves))
        if parent_leaves.get(key) != child_leaves.get(key)
    )


def build_host_mutation_delta(
    parent_policy: Mapping[str, Any],
    child_policy: Mapping[str, Any],
    *,
    attempt: int,
    mutation_level: str,
    parent_policy_sha256: str,
    child_policy_sha256: str,
) -> dict[str, Any]:
    """Build the canonical host delta that a formal trace must persist."""

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise PolicyProjectionError("mutation attempt must be a positive integer")
    if mutation_level not in {
        "strategy_mutation",
        "coordination_mutation",
        "parameter_tuning",
    }:
        raise PolicyProjectionError("unsupported mutation level")
    parent = executable_policy_projection(parent_policy)
    child = executable_policy_projection(child_policy)
    changed = changed_projection_dimensions(parent, child)
    return {
        "kind": "host_executable_mutation_delta",
        "schema_version": 1,
        "source": "host_executable_projection_diff",
        "attempt": attempt,
        "mutation_level": mutation_level,
        "parent_policy_sha256": parent_policy_sha256,
        "child_policy_sha256": child_policy_sha256,
        "parent_projection_sha256": projection_sha256(parent),
        "child_projection_sha256": projection_sha256(child),
        "changed_dimensions": list(changed),
        "effective": bool(changed),
    }


def strategy_matches_projection(
    strategy: object,
    projection: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Cross-check model-described strategy fields against policy bytes."""

    value = _mapping(strategy, label)
    committed = _mapping(projection.get("committed_subtask"), "projection.committed_subtask")
    for field in ("primitive", "semantic_objective", "coordination_mode"):
        actual = _required_text(value.get(field), f"{label}.{field}")
        if actual != committed.get(field):
            raise PolicyProjectionError(
                f"{label}.{field} does not match executable policy projection"
            )
    raw_team = _mapping(value.get("team_assignment"), f"{label}.team_assignment")
    projected_assignments = _mapping(
        projection.get("assignments"),
        "projection.assignments",
    )
    if set(raw_team) != set(projected_assignments):
        raise PolicyProjectionError(
            f"{label}.team_assignment robot IDs do not match executable policy"
        )
    for robot_id, raw in raw_team.items():
        item = _mapping(raw, f"{label}.team_assignment.{robot_id}")
        role = _required_text(item.get("role"), f"{label}.team_assignment.{robot_id}.role")
        projected = _mapping(
            projected_assignments[robot_id],
            f"projection.assignments.{robot_id}",
        )
        if role != projected.get("role"):
            raise PolicyProjectionError(
                f"{label}.team_assignment.{robot_id}.role does not match policy"
            )


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], child_prefix))
        return result
    return {prefix: value}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyProjectionError(f"{label} must be an object")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyProjectionError(f"{label} must be a non-empty string")
    return value.strip()


__all__ = [
    "PolicyProjectionError",
    "build_host_mutation_delta",
    "changed_projection_dimensions",
    "executable_policy_projection",
    "projection_sha256",
    "strategy_matches_projection",
]
