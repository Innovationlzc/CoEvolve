"""Provider-neutral structured-output schemas for model roles."""

from __future__ import annotations

from typing import Any


_STRING = {"type": "string"}
_STRING_ARRAY = {"type": "array", "items": _STRING}
_DIRECTION_VECTOR = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
}


def role_output_schema(role: str, context: dict[str, Any]) -> dict[str, Any]:
    """Return one logical JSON schema shared by every model provider."""

    if role == "task_orchestrator":
        return _object_schema(
            {
                "task_goal": _STRING,
                "global_plan": {
                    "type": "array",
                    "items": _phase_schema(),
                    "minItems": 1,
                },
                "current_phase_id": _STRING,
                "current_phase_status": {
                    "type": "string",
                    "enum": ["pending", "active", "completed"],
                },
                "current_subtask": _subtask_schema(),
                "constraints": _STRING_ARRAY,
                "rationale": _STRING,
            }
        )
    if role == "coder" and context.get("generation_scope") == "async_robot":
        return _object_schema(
            {
                "robot_id": _STRING,
                "script_id": _STRING,
                "role": _STRING,
                "objective": _STRING,
                "synchronization_group": _STRING,
                "phase_id": _STRING,
                "contact_intent": _STRING,
                "coordination_mode": {"type": "string", "enum": ["async"]},
                "code": _STRING,
            }
        )
    if role == "code_repair" and context.get("generation_scope") == "async_team_review":
        return _object_schema(
            {
                "decision": {
                    "type": "string",
                    "enum": ["approve_drafts", "repair_drafts"],
                },
                "conflicts": _STRING_ARRAY,
                "repaired_code": _object_schema(
                    {"front": _STRING, "back": _STRING}
                ),
                "rationale": _STRING,
            }
        )
    if role in {"coder", "code_repair"}:
        return _object_schema(
            {
                "script_id": _STRING,
                "plan": _STRING_ARRAY,
                "assignments": {
                    "type": "array",
                    "items": _robot_assignment_schema(executable=False),
                },
                "robot_scripts": {
                    "type": "array",
                    "items": _robot_assignment_schema(executable=True),
                },
                "rationale": _STRING,
                "spec_candidate": _STRING,
            }
        )
    if role == "vlm_failure_analyzer":
        return _object_schema(
            {
                "summary": _STRING,
                "scene_visible": {"type": "boolean"},
                "target_visible": {"type": "boolean"},
                "robots_standing_stably": {"type": "boolean"},
                "payload_on_floor": {"type": "boolean"},
                "payload_placed_on_target": {"type": "boolean"},
                "wall_collision_or_stuck": {"type": "boolean"},
                "failure_type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "payload_drop",
                        "robot_fall",
                        "payload_wall_contact",
                        "manual_review_required",
                    ],
                },
                "visible_evidence": _STRING_ARRAY,
                "relative_robot_layout": _STRING,
                "payload_axis_relative_to_passage": _STRING,
                "target_relation_to_team": _STRING,
                "visible_free_space_connectivity": _STRING_ARRAY,
                "passage_geometry_observations": _STRING_ARRAY,
                "team_heading_relative_to_target": {
                    "type": "string",
                    "enum": ["toward", "away", "crosswise", "mixed", "uncertain"],
                },
                "team_heading_relative_to_visible_free_space": {
                    "type": "string",
                    "enum": [
                        "aligned_with_open_passage",
                        "toward_wall_or_blocked_space",
                        "crosswise_to_passage",
                        "mixed",
                        "uncertain",
                    ],
                },
                "team_payload_posture_relative_to_visible_passage": {
                    "type": "string",
                    "enum": [
                        "longitudinal_aligned",
                        "lateral_aligned",
                        "diagonal_or_misaligned",
                        "mixed",
                        "uncertain",
                    ],
                },
                "team_footprint_clearance": {
                    "type": "string",
                    "enum": ["clear", "tight", "blocked", "uncertain"],
                },
                "immediate_route_direction_risk": {"type": "boolean"},
                "route_direction_evidence": _STRING,
                "route_direction_risk_basis": {
                    "type": "string",
                    "enum": [
                        "none",
                        "heading_only",
                        "executed_motion_into_blocked_space",
                        "unexplained_retreat",
                        "uncertain",
                    ],
                },
                "failure_analysis": _failure_analysis_schema(),
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            }
        )
    if role == "evolution_manager":
        return _object_schema(
            {
                "mutation_level": _mutation_level_schema(context),
                "failure_hypothesis": _STRING,
                "evidence": _STRING_ARRAY,
                "preserve": _STRING_ARRAY,
                "mutate": _STRING_ARRAY,
                "forbidden": _STRING_ARRAY,
                "success_test": _STRING,
                "inheritance": _STRING,
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            }
        )
    if role == "mutation_operator":
        return _object_schema(
            {
                "mutation_level": _mutation_level_schema(context),
                "reason": _STRING,
                "parent_strategy": _strategy_schema(include_team_assignment=True),
                "child_strategy": _strategy_schema(include_team_assignment=True),
                "changed_dimensions": _STRING_ARRAY,
                "inheritance": _STRING,
            }
        )
    if role == "memory_update":
        return _object_schema(
            {
                "promote": {"type": "boolean"},
                "strategy_name": _STRING,
                "memory_kind": _STRING,
                "mutation_level": _STRING,
                "applicability": _STRING_ARRAY,
                "parent_strategy": _STRING,
                "child_strategy": _STRING,
                "learned_rule": _STRING,
                "avoid": _STRING_ARRAY,
                "evidence_refs": _STRING_ARRAY,
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            }
        )
    return {"type": "object"}


def _mutation_level_schema(context: dict[str, Any]) -> dict[str, Any]:
    contract = context.get("mutation_contract", {})
    contract = contract if isinstance(contract, dict) else {}
    levels = contract.get("available_levels", ())
    levels = [str(item) for item in levels if str(item)]
    if not levels:
        levels = [
            "strategy_mutation",
            "coordination_mutation",
            "parameter_tuning",
        ]
    return {"type": "string", "enum": levels}


def _phase_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "phase_id": _STRING,
            "abstract_goal": _STRING,
            "completion_evidence": _STRING,
            "status": {
                "type": "string",
                "enum": ["pending", "active", "completed"],
            },
        }
    )


def _team_assignment_schema() -> dict[str, Any]:
    assignment = _object_schema({"role": _STRING, "objective": _STRING})
    return _object_schema({"front": assignment, "back": assignment})


def _strategy_schema(*, include_team_assignment: bool = False) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "primitive": _STRING,
        "semantic_objective": _STRING,
        "formation_strategy": _STRING,
        "coordination_mode": {
            "type": "string",
            "enum": ["sync", "async"],
        },
        "synchronization_intent": _STRING,
    }
    if include_team_assignment:
        fields["team_assignment"] = _team_assignment_schema()
    return _object_schema(fields)


def _subtask_schema() -> dict[str, Any]:
    properties = {
        "subtask_id": _STRING,
        "objective": _STRING,
        "guidance": _STRING,
        "strategy": _strategy_schema(),
        "setup_only": {"type": "boolean"},
        "navigation": _object_schema(
            {
                "reference": {
                    "type": "string",
                    "enum": [
                        "final_target",
                        "visible_connected_free_space",
                        "formation_only",
                        "payload_lifecycle",
                    ],
                },
                "arrival_condition": _STRING,
                "world_direction_xy": _DIRECTION_VECTOR,
                "completion_world_direction_xy": _DIRECTION_VECTOR,
            }
        ),
        "turn_comparison": _object_schema(
            {
                "active": {"type": "boolean"},
                "selected_direction": {
                    "type": "string",
                    "enum": ["none", "clockwise", "counterclockwise"],
                },
                "selected_heading": _STRING,
                "rejected_direction": {
                    "type": "string",
                    "enum": ["none", "clockwise", "counterclockwise"],
                },
                "rejected_heading": _STRING,
                "decision_basis": _STRING,
            }
        ),
        "team_assignment": _team_assignment_schema(),
        "completion_condition": _STRING,
        "preserve": _STRING_ARRAY,
    }
    return {
        "type": "object",
        "properties": properties,
        # OpenAI strict structured outputs require every declared property to
        # appear in ``required``.  ``setup_only`` is an explicit boolean
        # decision, not an omitted/optional transport field.
        "required": list(properties),
        "additionalProperties": False,
    }


def _robot_assignment_schema(*, executable: bool) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "robot_id": {"type": "string", "enum": ["front", "back"]},
        "role": _STRING,
        "objective": _STRING,
        "phase_id": _STRING,
        "contact_intent": _STRING,
        "synchronization_group": _STRING,
        "coordination_mode": {
            "type": "string",
            "enum": ["sync", "async"],
        },
    }
    if executable:
        fields.update({"script_id": _STRING, "code": _STRING})
    else:
        fields["safety_constraints"] = _STRING_ARRAY
    return _object_schema(fields)


def _failure_analysis_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "outcome": {
                "type": "string",
                "enum": ["accepted", "rejected"],
            },
            "primary_failure": _STRING,
            "causal_chain": _STRING_ARRAY,
            "physical_evidence": _STRING_ARRAY,
            "visual_evidence": _STRING_ARRAY,
            "strategy_defect": {"type": "boolean"},
            "coordination_defect": {"type": "boolean"},
            "magnitude_defect": {"type": "boolean"},
            "reusable_context": _STRING_ARRAY,
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        }
    )


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def validate_strict_json_schema(schema: dict[str, Any]) -> None:
    """Reject object schemas that provider strict mode would downgrade/reject."""

    def visit(node: object, path: str) -> None:
        if not isinstance(node, dict):
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            required = node.get("required")
            if not isinstance(required, list) or set(required) != set(properties):
                missing = sorted(set(properties) - set(required or ()))
                extra = sorted(set(required or ()) - set(properties))
                raise ValueError(
                    "strict JSON schema object must require every property at "
                    f"{path}; missing={missing} extra={extra}"
                )
            for key, child in properties.items():
                visit(child, f"{path}.properties.{key}")
        items = node.get("items")
        if items is not None:
            visit(items, f"{path}.items")
        for keyword in ("anyOf", "oneOf", "allOf"):
            variants = node.get(keyword)
            if isinstance(variants, list):
                for index, child in enumerate(variants):
                    visit(child, f"{path}.{keyword}[{index}]")

    visit(schema, "$")
