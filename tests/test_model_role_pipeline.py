from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from praxis_coop.cap_policy import (
    BoxCarryTask,
    EvaluationReport,
    ExecutionTrace,
    ModelCollaborativeEvolutionAgent,
    ModelRoleError,
    ModelRolePipeline,
    ModelTaskOrchestrator,
    ObservationBundle,
    PolicyScript,
    QwenVLBoxCarryVisualVerifier,
    RobotAssignment,
    RobotPolicyScript,
)
from praxis_coop.cap_policy.model_role_pipeline import (
    ModelRoleStallError,
    _parse_json_object,
)
from praxis_coop.cap_policy.local_navigation import (
    build_local_navigation_contract,
    validate_strategy_against_local_navigation,
)
from praxis_coop.cap_policy.generation_recovery import (
    NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL,
)
from praxis_coop.cap_policy.mutation_contract import (
    build_mutation_contract,
    host_effective_mutation_delta,
    validate_mutation_transition,
)
from praxis_coop.cap_policy.role_schemas import (
    role_output_schema,
    validate_strict_json_schema,
)
from praxis_coop.cap_policy.strategy_memory import (
    StrategyMemoryError,
    StrategyMemoryStore,
)
from praxis_coop.cap_policy.contracts import (
    RouteProgressGate,
    RouteSegment,
    VisualObservation,
)
from praxis_coop.cap_policy.model_agents import (
    ModelAgentError,
    _committed_route_recovery_contract,
    _fixed_global_camera_heading_projection,
    _generate_plan_guided_team_policy,
    _inherit_strategy_replan_navigation_contract,
    _loaded_linear_team_transaction_signature,
    _strategy_mutation_replan_contract,
    _validate_coordination_mutation_task_plan,
    _validate_orchestrator_plan_strategy,
    _validate_parameter_tuning_task_plan,
    _validate_plan_guided_mutation,
    _validate_selection_current_state_claims,
    _validate_strategy_mutation_orchestrator_replan,
)
from praxis_coop.configs import DEFAULT_MULTIMODAL_MODEL
from praxis_coop.llm import ProviderFatalError
from scripts.run_cap_box_carry_evolution import append_rollout_evidence


class FakeSharedQwenVL:
    model_name = DEFAULT_MULTIMODAL_MODEL

    def __init__(self, responses: list[dict]):
        self.responses = [json.dumps(item) for item in responses]
        self.call_records: list[dict] = []
        self.calls: list[dict] = []

    def generate(self, *, system: str, user: str) -> str:
        self.calls.append({"kind": "text", "system": system, "user": user})
        self.call_records.append(
            {
                "kind": "vision_text_generation",
                "input_tokens": 10,
                "output_tokens": 5,
                "latency_s": 0.1,
            }
        )
        return self.responses.pop(0)

    def analyze_image(
        self,
        *,
        system: str,
        user: str,
        image_path: str | Path,
    ) -> str:
        self.calls.append(
            {
                "kind": "vision",
                "system": system,
                "user": user,
                "image_path": str(image_path),
            }
        )
        self.call_records.append(
            {
                "kind": "vision_generation",
                "input_tokens": 20,
                "output_tokens": 8,
                "latency_s": 0.2,
            }
        )
        return self.responses.pop(0)


def test_parse_json_object_prefers_fenced_payload_after_prose_set_literal():
    raw = """Choose one distance from {1.0, 1.5, 2.0}.

```json
{"script_id": "bounded_transport", "distance_m": 1.0}
```
"""

    assert _parse_json_object(raw) == {
        "script_id": "bounded_transport",
        "distance_m": 1.0,
    }


def test_parse_json_object_skips_non_json_brace_before_plain_payload():
    raw = 'Allowed set {left, right}; result: {"direction": "left"}'

    assert _parse_json_object(raw) == {"direction": "left"}


def _task_plan() -> dict:
    return {
        "task_goal": "carry the payload to its target",
        "global_plan": [
            {
                "phase_id": "initial_straight",
                "abstract_goal": "transport through the initial passage",
                "completion_evidence": "global view shows the first corner reached",
                "status": "active",
            },
            {
                "phase_id": "first_corner",
                "abstract_goal": "enter the cross passage",
                "completion_evidence": "formation aligns with the cross passage",
                "status": "pending",
            },
        ],
        "current_phase_id": "initial_straight",
        "current_phase_status": "active",
        "current_subtask": {
            "subtask_id": "initial_straight_segment",
            "objective": "make one meaningful bounded passage advance",
            "guidance": "continue along the visibly connected passage",
            "navigation": {
                "reference": "visible_connected_free_space",
                "arrival_condition": "reach the next connected free-space frontier",
                "world_direction_xy": [0.0, 1.0],
                "completion_world_direction_xy": [0.0, 1.0],
            },
            "strategy": {
                "primitive": "coordinated_linear_transport",
                "coordination_mode": "sync",
                "semantic_objective": "forward",
                "formation_strategy": "side-by-side loaded transport",
                "synchronization_intent": "one identical atomic transaction",
            },
            "turn_comparison": {
                "active": False,
                "selected_direction": "none",
                "selected_heading": "none",
                "rejected_direction": "none",
                "rejected_heading": "none",
                "decision_basis": "not a signed turn",
            },
            "team_assignment": {
                robot_id: {
                    "role": "co-carrier",
                    "objective": "advance while preserving payload support",
                }
                for robot_id in ("front", "back")
            },
            "completion_condition": "safe measurable passage progress",
            "preserve": ["dual payload support"],
        },
        "constraints": ["preserve dual support"],
        "rationale": "the connected passage is visually clear",
    }


def _set_turn_comparison(
    plan: dict,
    *,
    direction: str,
    selected_heading: str,
    rejected_heading: str,
    decision_basis: str = "selected heading follows the connected passage",
) -> None:
    alternative = "counterclockwise" if direction == "clockwise" else "clockwise"
    plan["current_subtask"]["turn_comparison"] = {
        "active": True,
        "selected_direction": direction,
        "selected_heading": selected_heading,
        "rejected_direction": alternative,
        "rejected_heading": rejected_heading,
        "decision_basis": decision_basis,
    }


def _task_for_plan(plan: dict, **kwargs) -> BoxCarryTask:
    """Build scene data whose topology exactly matches a test plan fixture."""

    subtask = plan.get("current_subtask", {})
    subtask = subtask if isinstance(subtask, dict) else {}
    navigation = subtask.get("navigation", {})
    navigation = navigation if isinstance(navigation, dict) else {}
    raw_direction = navigation.get("world_direction_xy", [1.0, 0.0])
    try:
        direction_x = float(raw_direction[0])
        direction_y = float(raw_direction[1])
    except (IndexError, TypeError, ValueError):
        direction_x, direction_y = 1.0, 0.0
    if abs(direction_x) + abs(direction_y) <= 1.0e-9:
        direction_x, direction_y = 1.0, 0.0
    route_segments = kwargs.pop(
        "route_segments",
        (
            RouteSegment(
                segment_id="synthetic_plan_segment",
                direction_x=direction_x,
                direction_y=direction_y,
                start_x_m=0.0,
                start_y_m=0.0,
                goal_x_m=10.0 * direction_x,
                goal_y_m=10.0 * direction_y,
                center_x_m=5.0 * direction_x,
                center_y_m=5.0 * direction_y,
                half_width_m=10.0,
            ),
        ),
    )
    route_progress_gates = kwargs.pop(
        "route_progress_gates",
        (
            RouteProgressGate(
                phase_name="synthetic_plan_gate",
                normal_x=direction_x,
                normal_y=direction_y,
                threshold_m=9.0,
            ),
        ),
    )
    return BoxCarryTask(
        task_id=str(kwargs.pop("task_id", "plan_fixture")),
        target=str(kwargs.pop("target", "place payload on target")),
        mission_topology=tuple(
            str(phase["abstract_goal"])
            for phase in plan.get("global_plan", ())
        ),
        route_segments=route_segments,
        route_progress_gates=route_progress_gates,
        **kwargs,
    )


def _stale_payload_initialization_plan() -> dict:
    plan = _task_plan()
    plan["current_subtask"] = {
        "subtask_id": "stale_payload_initialization",
        "objective": "activate the deferred payload",
        "guidance": "repeat the payload initialization event",
        "strategy": {
            "primitive": "place_payload_on_arms",
            "coordination_mode": "sync",
            "semantic_objective": "place_payload_on_arms(hold_s=4)",
            "formation_strategy": "synchronized payload activation",
            "synchronization_intent": "one identical atomic transaction",
        },
        "team_assignment": {
            robot_id: {
                "role": "payload_support",
                "objective": "place the payload on the robot arms",
            }
            for robot_id in ("front", "back")
        },
        "completion_condition": "payload active with dual support",
        "preserve": ["upright robots"],
    }
    plan["rationale"] = "the stale previous plan says the payload is deferred"
    return plan


def _failure_analysis() -> dict:
    return {
        "summary": "The payload stayed supported but the robots diverged.",
        "scene_visible": True,
        "payload_on_floor": False,
        "robots_standing_stably": True,
        "wall_collision_or_stuck": False,
        "payload_placed_on_target": False,
        "visible_evidence": ["Both robots remain upright."],
        "target_visible": True,
        "target_relation_to_team": "visible beyond the connected passages",
        "team_heading_relative_to_target": "crosswise",
        "team_heading_relative_to_visible_free_space": (
            "aligned_with_open_passage"
        ),
        "team_payload_posture_relative_to_visible_passage": (
            "longitudinal_aligned"
        ),
        "visible_free_space_connectivity": [
            "the current passage remains visibly open"
        ],
        "passage_geometry_observations": [
            "the team remains inside the current passage"
        ],
        "team_footprint_clearance": "clear",
        "relative_robot_layout": "the two robots remain side by side",
        "payload_axis_relative_to_passage": "parallel to the visible passage",
        "immediate_route_direction_risk": False,
        "route_direction_risk_basis": "none",
        "route_direction_evidence": "",
        "failure_type": "none",
        "confidence": 0.9,
        "failure_analysis": {
            "outcome": "rejected",
            "primary_failure": "formation_desync",
            "causal_chain": ["equal commands produced unequal measured progress"],
            "physical_evidence": ["front/back displacement diverged"],
            "visual_evidence": ["the formation became staggered"],
            "strategy_defect": False,
            "coordination_defect": True,
            "magnitude_defect": False,
            "reusable_context": ["loaded linear transport"],
            "confidence": 0.9,
        },
    }


def _manager(level: str = "coordination_mutation") -> dict:
    return {
        "mutation_level": level,
        "failure_hypothesis": "the temporary coordination relation was unsuitable",
        "evidence": ["measured front/back displacement diverged"],
        "preserve": ["linear transport objective"],
        "mutate": ["front/back coordination relation"],
        "forbidden": ["parameter-only retry"],
        "success_test": "dual support with bounded relative displacement",
        "inheritance": "retain the verified motion objective",
        "confidence": 0.9,
    }


def _memory_update(*, promote: bool = True) -> dict:
    return {
        "promote": promote,
        "strategy_name": "asymmetric loaded transport recovery",
        "memory_kind": "coordination_genome",
        "mutation_level": "coordination_mutation",
        "applicability": ["rigid payload transport with measured role divergence"],
        "parent_strategy": "symmetric coordinated linear transport",
        "child_strategy": "role-aware coordinated linear transport",
        "learned_rule": (
            "preserve the route objective while changing the temporary role relation"
        ),
        "avoid": ["do not misclassify role divergence as a route failure"],
        "evidence_refs": ["verified rollout"],
        "confidence": 0.8,
    }


def _model_policy_payload() -> dict:
    code = (
        "def policy(api):\n"
        "    api.hold_position(hold_s=1.0)\n"
        "    api.stop_all()\n"
    )
    return {
        "script_id": "attempt_01",
        "code": code,
        "robot_scripts": {
            robot_id: {
                "script_id": f"{robot_id}_01",
                "code": code,
                "synchronization_group": "carry_team",
                "phase_id": "stabilize",
                "contact_intent": "maintain_payload_support",
                "coordination_mode": "sync",
            }
            for robot_id in ("front", "back")
        },
    }


def _async_robot_draft(robot_id: str) -> dict:
    delta = 0.0 if robot_id == "front" else 0.1
    code = (
        "def policy(api):\n"
        "    api.coordinated_alignment("
        f"axis='longitudinal', delta={delta}, duration_s=2.0)\n"
        "    api.stop_all()\n"
    )
    return {
        "robot_id": robot_id,
        "script_id": f"{robot_id}_async",
        "role": "reference" if robot_id == "front" else "moving",
        "objective": "restore loaded formation alignment",
        "phase_id": "alignment",
        "contact_intent": "maintain_payload_support",
        "synchronization_group": "carry_team",
        "coordination_mode": "async",
        "code": code,
    }


def _async_team_policy_payload() -> dict:
    drafts = {
        robot_id: _async_robot_draft(robot_id)
        for robot_id in ("front", "back")
    }
    return {
        "script_id": "async_team",
        "code": drafts["front"]["code"],
        "robot_scripts": {
            robot_id: {
                key: value
                for key, value in draft.items()
                if key not in {"role", "objective"}
            }
            for robot_id, draft in drafts.items()
        },
    }


def _async_team_review_payload() -> dict:
    return {
        "decision": "approve_drafts",
        "conflicts": [],
        "repaired_code": {"front": "", "back": ""},
        "rationale": "the two drafts implement one coherent async transaction",
    }


def _sync_linear_team_policy_payload() -> dict:
    code = (
        "def policy(api):\n"
        "    api.coordinated_linear_transport("
        "direction='forward', distance_m=1.0, speed_mps=0.5, ramp_s=1.5)\n"
        "    api.stop_all()\n"
    )
    assignments = {
        robot_id: {
            "robot_id": robot_id,
            "role": "co-carrier",
            "objective": "advance while preserving payload support",
            "synchronization_group": "carry_team",
            "safety_constraints": ["preserve payload support"],
            "phase_id": "initial_straight_segment",
            "contact_intent": "maintain_payload_support",
            "coordination_mode": "sync",
        }
        for robot_id in ("front", "back")
    }
    scripts = {
        robot_id: {
            "robot_id": robot_id,
            "script_id": f"{robot_id}_linear",
            "code": code,
            "synchronization_group": "carry_team",
            "phase_id": "initial_straight_segment",
            "contact_intent": "maintain_payload_support",
            "coordination_mode": "sync",
        }
        for robot_id in ("front", "back")
    }
    return {
        "script_id": "sync_linear_team",
        "assignments": assignments,
        "robot_scripts": scripts,
        "rationale": "execute the committed straight-passage subtask",
        "plan": ["advance through the active passage"],
        "spec_candidate": "",
    }


def _mutation() -> dict:
    return {
        "mutation_level": "coordination_mutation",
        "reason": "preserve direction and change temporary roles",
        "parent_strategy": {
            "primitive": "coordinated_linear_transport",
            "semantic_objective": "forward",
            "formation_strategy": "side-by-side loaded transport",
            "coordination_mode": "sync",
        },
        "child_strategy": {
            "primitive": "coordinated_linear_transport",
            "semantic_objective": "forward",
            "formation_strategy": "side-by-side loaded transport",
            "coordination_mode": "async",
            "synchronization_intent": "atomic role-aware transport",
            "team_assignment": {
                "front": {
                    "role": "reference carrier",
                    "objective": "maintain the shared passage objective",
                },
                "back": {
                    "role": "compensating carrier",
                    "objective": "match measured team progress",
                },
            },
        },
        "changed_dimensions": ["temporary roles"],
        "inheritance": "preserve the linear objective",
    }


def _pivot_strategy_mutation() -> dict:
    return {
        "mutation_level": "strategy_mutation",
        "reason": "the current subtask requires a shared heading change",
        "parent_strategy": _task_plan()["current_subtask"]["strategy"],
        "child_strategy": {
            "primitive": "coordinated_pivot",
            "semantic_objective": "counterclockwise",
            "formation_strategy": "stationary inner support with outer forward swing",
            "coordination_mode": "async",
            "synchronization_intent": "one atomic inner-outer turn transaction",
            "team_assignment": {
                "front": {
                    "role": "inner",
                    "objective": "maintain the pivot support",
                },
                "back": {
                    "role": "outer",
                    "objective": "swing forward around the inner support",
                },
            },
        },
        "changed_dimensions": [
            "maneuver primitive",
            "semantic objective",
            "formation strategy",
        ],
        "inheritance": "preserve the active route phase and payload support",
    }


def _pivot_task_plan() -> dict:
    plan = _task_plan()
    child = _pivot_strategy_mutation()["child_strategy"]
    plan["current_subtask"] = {
        "subtask_id": "counterclockwise_corner_pivot",
        "objective": "execute one bounded counterclockwise pivot",
        "guidance": "turn counterclockwise while preserving bilateral support",
        "navigation": {
            "reference": "visible_connected_free_space",
            "arrival_condition": "align with the next connected passage",
            "world_direction_xy": [0.0, 1.0],
            "completion_world_direction_xy": [0.0, 1.0],
        },
        "strategy": {
            key: value
            for key, value in child.items()
            if key != "team_assignment"
        },
        "team_assignment": child["team_assignment"],
        "completion_condition": (
            "counterclockwise heading change is visible and support is retained"
        ),
        "preserve": ["dual payload support"],
    }
    _set_turn_comparison(
        plan,
        direction="counterclockwise",
        selected_heading="down-left",
        rejected_heading="up-left",
    )
    plan["rationale"] = (
        "a counterclockwise pivot is the approved geometry-changing maneuver"
    )
    return plan


def _pivot_robot_draft(robot_id: str) -> dict:
    role = "inner" if robot_id == "front" else "outer"
    code = (
        "def policy(api):\n"
        "    api.coordinated_pivot("
        "turn_direction='counterclockwise', angle_deg=30.0, "
        "travel_direction='forward', duration_s=8.0, "
        f"role='{role}', swing_radius_m=0.8)\n"
        "    api.stop_all()\n"
    )
    return {
        "robot_id": robot_id,
        "script_id": f"{robot_id}_pivot",
        "role": role,
        "objective": (
            "maintain the pivot support"
            if role == "inner"
            else "swing forward around the inner support"
        ),
        "phase_id": "first_corner_turn",
        "contact_intent": "maintain_payload_support",
        "synchronization_group": "corner_team",
        "coordination_mode": "async",
        "code": code,
    }


def _pivot_team_policy_payload() -> dict:
    drafts = {
        robot_id: _pivot_robot_draft(robot_id)
        for robot_id in ("front", "back")
    }
    return {
        "script_id": "async_pivot_team",
        "assignments": {
            robot_id: {
                "robot_id": robot_id,
                "role": draft["role"],
                "objective": draft["objective"],
                "synchronization_group": "corner_team",
                "safety_constraints": ["preserve payload support"],
                "phase_id": "first_corner_turn",
                "contact_intent": "maintain_payload_support",
                "coordination_mode": "async",
            }
            for robot_id, draft in drafts.items()
        },
        "robot_scripts": {
            robot_id: {
                key: value
                for key, value in draft.items()
                if key not in {"role", "objective"}
            }
            for robot_id, draft in drafts.items()
        },
        "rationale": "execute the evolved corner strategy",
        "plan": ["perform one bounded shared heading change"],
        "spec_candidate": "",
    }


def _policy(*, distance_m: float = 1.0) -> PolicyScript:
    code = (
        "def policy(api):\n"
        "    api.coordinated_linear_transport("
        f"direction='forward', distance_m={distance_m}, speed_mps=0.5, ramp_s=1.5)\n"
        "    api.stop_all()\n"
    )
    return PolicyScript(
        script_id="attempt_01",
        attempt=1,
        code=code,
        robot_scripts=tuple(
            RobotPolicyScript(
                robot_id=robot_id,
                script_id=f"{robot_id}_01",
                code=code,
                synchronization_group="carry_team",
                phase_id="transport",
                contact_intent="maintain_payload_support",
                coordination_mode="sync",
            )
            for robot_id in ("front", "back")
        ),
    )


def test_role_pipeline_uses_one_shared_qwen_and_writes_separate_artifacts(tmp_path):
    frame = tmp_path / "global.png"
    frame.write_bytes(b"frame")
    client = FakeSharedQwenVL(
        [
            _task_plan(),
            _model_policy_payload(),
            _failure_analysis(),
            _manager(),
            _mutation(),
            _model_policy_payload(),
            _memory_update(),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    pipeline.plan_task({}, artifact_id="attempt_01", image_path=frame)
    pipeline.code_policy({}, artifact_id="attempt_01")
    pipeline.analyze_rollout({}, artifact_id="attempt_01", image_path=frame)
    pipeline.decide_mutation({}, artifact_id="attempt_01", image_path=frame)
    pipeline.mutate_strategy({}, artifact_id="attempt_01")
    pipeline.repair_code({}, artifact_id="attempt_01")
    pipeline.propose_memory_update({}, artifact_id="attempt_01")

    assert pipeline.client is client
    assert {record["role"] for record in pipeline.records} == {
        "task_orchestrator",
        "coder",
        "vlm_failure_analyzer",
        "evolution_manager",
        "mutation_operator",
        "code_repair",
        "memory_update",
    }
    assert all(record["model_name"] == DEFAULT_MULTIMODAL_MODEL for record in pipeline.records)
    assert (tmp_path / "model_roles" / "task_orchestrator_attempt_01.json").exists()
    assert (tmp_path / "model_roles" / "code_repair_attempt_01.json").exists()
    assert (tmp_path / "model_roles" / "memory_update_attempt_01.json").exists()
    usage = pipeline.usage_summary()
    assert usage["shared_model_instance"] is True
    assert usage["roles"]["vlm_failure_analyzer"]["call_count"] == 1


def test_memory_update_normalizes_controller_level_alias(tmp_path):
    proposal = _memory_update()
    proposal["mutation_level"] = "strategy"
    client = FakeSharedQwenVL([proposal])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    normalized = pipeline.propose_memory_update(
        {},
        artifact_id="attempt_01",
    )

    assert normalized["mutation_level"] == "strategy_mutation"
    artifact = json.loads(
        (
            tmp_path
            / "model_roles"
            / "memory_update_attempt_01.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["host_schema_normalizations"] == [
        "mutation_level: strategy -> strategy_mutation"
    ]


def test_task_orchestrator_normalizes_rollout_outcome_status_aliases(tmp_path):
    plan = _task_plan()
    plan["global_plan"][0]["status"] = "failed"
    plan["global_plan"][1]["status"] = "not_started"
    plan["current_phase_status"] = "failed"
    plan["current_subtask"]["strategy"]["coordination_mode"] = "sync_only"
    pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([plan]),
        output_dir=tmp_path,
        response_retries=0,
    )

    normalized = pipeline.plan_task({}, artifact_id="attempt_04")

    assert normalized["global_plan"][0]["status"] == "active"
    assert normalized["global_plan"][1]["status"] == "pending"
    assert normalized["current_phase_status"] == "active"
    assert normalized["current_subtask"]["strategy"]["coordination_mode"] == "sync"
    record = pipeline.records[0]
    assert "global_plan[0].status: failed -> active" in record[
        "host_schema_normalizations"
    ]
    assert (
        "current_subtask.strategy.coordination_mode: sync_only -> sync"
        in record["host_schema_normalizations"]
    )


def test_task_orchestrator_retry_names_exact_sdk_vocabulary(tmp_path):
    invalid = _task_plan()
    invalid["current_subtask"]["strategy"]["primitive"] = "carry_walk"
    valid = _task_plan()
    valid["current_subtask"]["strategy"]["primitive"] = "carry_forward"
    client = FakeSharedQwenVL([invalid, valid])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def validate_sdk(plan):
        primitive = plan["current_subtask"]["strategy"]["primitive"]
        if primitive != "carry_forward":
            raise ModelRoleError(
                "current_subtask.strategy.primitive must be a safe SDK method; "
                f"got {primitive!r}; allowed=['carry_forward']"
            )

    result = pipeline.plan_task(
        {"exact_sdk_primitive_names": ["carry_forward"]},
        artifact_id="attempt_04",
        validator=validate_sdk,
    )

    assert result["current_subtask"]["strategy"]["primitive"] == "carry_forward"
    retry_prompt = client.calls[1]["user"]
    assert "one exact unchanged token" in retry_prompt
    assert "['carry_forward']" in retry_prompt


def test_task_orchestrator_rejects_unresolvable_runtime_local_goal(tmp_path):
    invalid = _task_plan()
    invalid["current_subtask"]["navigation"]["reference"] = "runtime_local_goal"
    valid = _task_plan()
    client = FakeSharedQwenVL([invalid, valid])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.plan_task({}, artifact_id="attempt_08")

    assert result["current_subtask"]["navigation"]["reference"] == (
        "visible_connected_free_space"
    )
    assert pipeline.records[0]["valid"] is False
    assert "not host-resolvable" in pipeline.records[0]["validation_error"]


def test_task_orchestrator_rejects_conflicting_route_completion_direction(tmp_path):
    invalid = _task_plan()
    invalid["current_subtask"]["navigation"][
        "completion_world_direction_xy"
    ] = [1.0, 0.0]
    valid = _task_plan()
    client = FakeSharedQwenVL([invalid, valid])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.plan_task({}, artifact_id="attempt_08")

    assert result["current_subtask"]["navigation"][
        "completion_world_direction_xy"
    ] == [0.0, 1.0]
    assert "STRUCTURED_CONTRACT_CONFLICT" in pipeline.records[0][
        "validation_error"
    ]


def test_task_orchestrator_retry_replaces_point_target_backward_with_pivot(tmp_path):
    runtime_state = {
        "checkpoint": {"checkpoint_id": "accepted_step_16"},
        "payload": {
            "active": True,
            "x": 0.6303562549231021,
            "y": 7.769384449186844,
        },
        "target": {
            "available": True,
            "center_x_m": -1.5,
            "center_y_m": 7.6,
        },
        "robots": {
            "front": {"yaw_deg": -11.7706},
            "back": {"yaw_deg": -10.9781},
        },
    }
    invalid = _task_plan()
    invalid["current_subtask"]["navigation"] = {
        "reference": "final_target",
        "arrival_condition": "reach the final target",
        "world_direction_xy": [0, 0],
    }
    invalid["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_linear_transport",
            "semantic_objective": "backward",
            "coordination_mode": "sync",
        }
    )
    corrected = json.loads(json.dumps(invalid))
    corrected["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_pivot",
            "semantic_objective": "clockwise",
            "coordination_mode": "async",
        }
    )
    corrected["current_subtask"]["team_assignment"] = {
        "front": {
            "role": "inner",
            "objective": "maintain the inner pivot support",
        },
        "back": {
            "role": "outer",
            "objective": "advance on the outer pivot arc",
        },
    }
    _set_turn_comparison(
        corrected,
        direction="clockwise",
        selected_heading="left",
        rejected_heading="right",
    )
    client = FakeSharedQwenVL([invalid, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def validate_navigation(plan):
        strategy = plan["current_subtask"]["strategy"]
        contract = build_local_navigation_contract(plan, runtime_state)
        validate_strategy_against_local_navigation(
            method=strategy["primitive"],
            objective=strategy["semantic_objective"],
            contract=contract,
        )

    result = pipeline.plan_task(
        {"local_navigation_contract": "derived from the trusted live state"},
        artifact_id="attempt_24",
        validator=validate_navigation,
    )

    assert result["current_subtask"]["strategy"]["primitive"] == (
        "coordinated_pivot"
    )
    retry_prompt = client.calls[1]["user"]
    assert "point-reference heading contract" in retry_prompt
    assert "select coordinated_pivot" in retry_prompt
    assert "Do not emit forward, backward, left, or right" in retry_prompt
    assert "Select one of ['backward']" not in retry_prompt


def test_payload_activation_cannot_complete_initial_route_phase(tmp_path):
    previous = _task_plan()
    invalid = json.loads(json.dumps(previous))
    invalid["global_plan"][0]["status"] = "completed"
    invalid["global_plan"][1]["status"] = "active"
    invalid["current_phase_id"] = "first_corner"
    invalid["current_phase_status"] = "active"
    corrected = _task_plan()
    client = FakeSharedQwenVL([invalid, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task(
        {
            "last_accepted_policy": {
                "robot_scripts": [
                    {
                        "robot_id": "front",
                        "code": (
                            "def policy(api):\n"
                            "    api.place_payload_on_arms(hold_s=4)\n"
                            "    api.stop_all()"
                        ),
                    }
                ]
            },
            "last_accepted_rollout": {"success": True},
        },
        artifact_id="attempt_02",
    )

    assert result["current_phase_id"] == "initial_straight"
    first_record = pipeline.records[0]
    assert first_record["valid"] is False
    assert "cannot complete a route phase" in first_record["validation_error"]
    assert "keep the previous phase active" in client.calls[1]["user"]


def test_persistent_phase_id_aliases_are_normalized_when_goals_match(tmp_path):
    previous = _task_plan()
    renamed = json.loads(json.dumps(previous))
    renamed["global_plan"][0]["phase_id"] = "transport_initial_passage"
    renamed["global_plan"][1]["phase_id"] = "negotiate_first_corner"
    renamed["current_phase_id"] = "transport_initial_passage"
    client = FakeSharedQwenVL([renamed])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task({}, artifact_id="attempt_02")

    assert [phase["phase_id"] for phase in result["global_plan"]] == [
        "initial_straight",
        "first_corner",
    ]
    assert result["current_phase_id"] == "initial_straight"
    assert pipeline.records[0]["host_schema_normalizations"] == [
        "global_plan backbone: restored persistent IDs and descriptions by "
        "unchanged semantic route-phase order"
    ]


def test_persistent_phase_backbone_accepts_equivalent_goal_prose(tmp_path):
    previous = _task_plan()
    rephrased = json.loads(json.dumps(previous))
    rephrased["global_plan"][0].update(
        {
            "phase_id": "phase_1_initial_straight",
            "abstract_goal": "Traverse the initial straight passage.",
            "completion_evidence": "Fresh route evidence proves traversal.",
        }
    )
    rephrased["global_plan"][1].update(
        {
            "phase_id": "phase_2_first_corner_to_cross",
            "abstract_goal": "Negotiate the first corner into the cross passage.",
            "completion_evidence": "Fresh route evidence proves the turn.",
        }
    )
    rephrased["current_phase_id"] = "phase_1_initial_straight"
    client = FakeSharedQwenVL([rephrased])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task({}, artifact_id="attempt_02")

    assert [phase["phase_id"] for phase in result["global_plan"]] == [
        "initial_straight",
        "first_corner",
    ]
    assert result["global_plan"][0]["abstract_goal"] == (
        previous["global_plan"][0]["abstract_goal"]
    )
    assert result["global_plan"][0]["completion_evidence"] == (
        previous["global_plan"][0]["completion_evidence"]
    )


def test_role_pipeline_resume_preserves_call_history_and_suffixes(tmp_path):
    client = FakeSharedQwenVL([])
    first = ModelRolePipeline(client=client, output_dir=tmp_path)
    first._record_call(
        {
            "role": "task_orchestrator",
            "artifact_id": "attempt_02",
            "valid": False,
        }
    )

    resumed = ModelRolePipeline(client=client, output_dir=tmp_path)
    resumed._record_call(
        {
            "role": "task_orchestrator",
            "artifact_id": "attempt_02",
            "valid": True,
        }
    )

    assert len(resumed.records) == 2
    assert (
        tmp_path
        / "model_roles"
        / "task_orchestrator_attempt_02_call_02.json"
    ).exists()
    history = json.loads(
        (tmp_path / "model_roles" / "role_call_index.json").read_text()
    )
    assert [item["valid"] for item in history["calls"]] == [False, True]


def test_persistent_phase_id_aliases_do_not_hide_reordered_goals(tmp_path):
    previous = _task_plan()
    reordered = json.loads(json.dumps(previous))
    reordered["global_plan"].reverse()
    reordered["global_plan"][0]["phase_id"] = "renamed_first"
    reordered["global_plan"][1]["phase_id"] = "renamed_second"
    reordered["current_phase_id"] = "renamed_second"
    corrected = _task_plan()
    client = FakeSharedQwenVL([reordered, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task({}, artifact_id="attempt_02")

    assert result["current_phase_id"] == "initial_straight"
    assert pipeline.records[0]["valid"] is False
    assert "persistent route backbone" in pipeline.records[0]["validation_error"]


def test_accepted_route_motion_may_advance_one_phase(tmp_path):
    previous = _task_plan()
    advanced = json.loads(json.dumps(previous))
    advanced["global_plan"][0]["status"] = "completed"
    advanced["global_plan"][1]["status"] = "active"
    advanced["current_phase_id"] = "first_corner"
    advanced["current_phase_status"] = "active"
    client = FakeSharedQwenVL([advanced])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task(
        {
            "last_accepted_policy": {
                "robot_scripts": [
                    {
                        "robot_id": "front",
                        "code": (
                            "def policy(api):\n"
                            "    api.coordinated_linear_transport("
                            "direction='forward', distance_m=2.0, "
                            "speed_mps=0.5, ramp_s=1.5)\n"
                            "    api.stop_all()"
                        ),
                    }
                ]
            },
            "last_accepted_rollout": {"success": True},
        },
        artifact_id="attempt_03",
    )

    assert result["current_phase_id"] == "first_corner"
    assert result["global_plan"][0]["status"] == "completed"


def test_accepted_history_may_catch_up_multiple_contiguous_phases(tmp_path):
    previous = _task_plan()
    previous["global_plan"].append(
        {
            "phase_id": "cross_passage",
            "abstract_goal": "traverse the cross passage",
            "completion_evidence": "global view shows the second corner reached",
            "status": "pending",
        }
    )
    advanced = json.loads(json.dumps(previous))
    advanced["global_plan"][0]["status"] = "completed"
    advanced["global_plan"][1]["status"] = "completed"
    advanced["global_plan"][2]["status"] = "active"
    advanced["current_phase_id"] = "cross_passage"
    advanced["current_phase_status"] = "active"
    client = FakeSharedQwenVL([advanced])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task(
        {
            "last_accepted_policy": {
                "robot_scripts": [
                    {
                        "robot_id": "front",
                        "code": (
                            "def policy(api):\n"
                            "    api.coordinated_pivot("
                            "turn_direction='counterclockwise', angle_deg=30, "
                            "travel_direction='forward', duration_s=8, "
                            "role='outer', swing_radius_m=0.8)\n"
                            "    api.stop_all()"
                        ),
                    }
                ]
            },
            "last_accepted_rollout": {"success": True},
        },
        artifact_id="attempt_06",
    )

    assert result["current_phase_id"] == "cross_passage"
    assert [phase["status"] for phase in result["global_plan"]] == [
        "completed",
        "completed",
        "active",
    ]


def test_trusted_checkpoint_advances_stale_plan_to_physical_frontier(tmp_path):
    previous = _task_plan()
    stale_response = json.loads(json.dumps(previous))
    client = FakeSharedQwenVL([stale_response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task(
        {
            "accepted_route_action_history": {
                "latest_checkpoint": "accepted_step_06",
                "steps": [
                    {
                        "checkpoint_after": "accepted_step_06",
                        "method": "coordinated_linear_transport",
                        "semantic_objective": "forward",
                    }
                ],
            },
            "current_trusted_observation": {
                "runtime_state": {
                    "checkpoint": {"checkpoint_id": "accepted_step_06"}
                }
            },
            "host_route_phase_guard": {
                "source": "trusted_checkpoint_physics",
                "current_topological_phase": "first_corner",
                "maximum_completed_prefix": 1,
                "coordinates_hidden_from_model": True,
            },
        },
        artifact_id="attempt_07",
    )

    assert result["current_phase_id"] == "first_corner"
    assert [phase["status"] for phase in result["global_plan"]] == [
        "completed",
        "active",
    ]
    assert pipeline.records[0]["host_schema_normalizations"] == [
        "global_plan statuses: aligned the persistent completion prefix to the "
        "trusted physical frontier 1 and restored one current phase"
    ]


def test_mission_topology_rejects_inserted_lifecycle_route_phase(tmp_path):
    mission_topology = [
        "cross the loading area",
        "turn into the transfer aisle",
        "place and release the payload on the destination",
    ]
    invalid = _task_plan()
    invalid["global_plan"] = [
        {
            "phase_id": "payload_bootstrap",
            "abstract_goal": "place payload on arms",
            "completion_evidence": "payload is supported",
            "status": "active",
        },
        *[
            {
                "phase_id": f"phase_{index}",
                "abstract_goal": goal,
                "completion_evidence": "trusted physical transition",
                "status": "pending",
            }
            for index, goal in enumerate(mission_topology)
        ],
    ]
    invalid["current_phase_id"] = "payload_bootstrap"

    corrected = json.loads(json.dumps(invalid))
    corrected["global_plan"] = corrected["global_plan"][1:]
    corrected["global_plan"][0]["status"] = "active"
    corrected["current_phase_id"] = "phase_0"

    client = FakeSharedQwenVL([invalid, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.plan_task(
        {"mission_topology": mission_topology},
        artifact_id="attempt_01",
    )

    assert len(client.calls) == 2
    assert [phase["abstract_goal"] for phase in result["global_plan"]] == (
        mission_topology
    )
    assert result["current_phase_id"] == "phase_0"


def test_rotated_scene_topology_uses_trusted_frontier_without_named_phases(tmp_path):
    mission_topology = [
        "traverse segment alpha",
        "change heading at junction beta",
        "approach translated destination gamma",
    ]
    previous = _task_plan()
    previous["global_plan"] = [
        {
            "phase_id": f"arbitrary_{index}",
            "abstract_goal": goal,
            "completion_evidence": "trusted physical transition",
            "status": "active" if index == 0 else "pending",
        }
        for index, goal in enumerate(mission_topology)
    ]
    previous["current_phase_id"] = "arbitrary_0"
    stale_response = json.loads(json.dumps(previous))

    client = FakeSharedQwenVL([stale_response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous
    result = pipeline.plan_task(
        {
            "mission_topology": mission_topology,
            "accepted_route_action_history": {
                "latest_checkpoint": "accepted_step_08",
                "steps": [
                    {"checkpoint_after": "accepted_step_08"},
                ],
            },
            "current_trusted_observation": {
                "runtime_state": {
                    "checkpoint": {"checkpoint_id": "accepted_step_08"}
                }
            },
            "host_route_phase_guard": {
                "source": "trusted_checkpoint_physics",
                "current_topological_phase": "translated_gate_beta",
                "maximum_completed_prefix": 2,
                "coordinates_hidden_from_model": True,
            },
        },
        artifact_id="attempt_09",
    )

    assert result["current_phase_id"] == "arbitrary_2"
    assert [phase["status"] for phase in result["global_plan"]] == [
        "completed",
        "completed",
        "active",
    ]
    assert "clockwise" not in json.dumps(result)


def test_physical_frontier_refreshes_final_target_navigation_after_phase_update(
    tmp_path,
):
    task = BoxCarryTask(task_id="generic_final_approach", target="place payload")
    phases = [
        {
            "phase_id": f"phase_{index}",
            "abstract_goal": goal,
            "completion_evidence": "trusted physical transition",
            "status": "active" if index == 0 else "pending",
        }
        for index, goal in enumerate(task.mission_topology)
    ]
    stale = _task_plan()
    stale["global_plan"] = phases
    stale["current_phase_id"] = "phase_0"

    corrected = json.loads(json.dumps(stale))
    corrected["current_subtask"]["navigation"] = {
        "reference": "final_target",
        "arrival_condition": "reach the declared target",
        "world_direction_xy": [0.0, 0.0],
    }

    client = FakeSharedQwenVL([stale, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    result = pipeline.plan_task(
        {
            "mission_topology": list(task.mission_topology),
            "accepted_route_action_history": {
                "latest_checkpoint": "accepted_step_17",
                "steps": [{"checkpoint_after": "accepted_step_17"}],
            },
            "current_trusted_observation": {
                "runtime_state": {
                    "checkpoint": {"checkpoint_id": "accepted_step_17"}
                }
            },
            "host_route_phase_guard": {
                "source": "trusted_checkpoint_physics",
                "current_topological_phase": "translated_final_gate",
                "maximum_completed_prefix": 4,
                "coordinates_hidden_from_model": True,
            },
        },
        artifact_id="attempt_18",
    )

    assert len(client.calls) == 2
    assert result["current_phase_id"] == "phase_4"
    assert result["global_plan"][4]["abstract_goal"] == (
        task.mission_topology[4]
    )
    assert result["current_subtask"]["navigation"]["reference"] == "final_target"


def test_accepted_route_history_cannot_complete_phase_without_physical_guard(tmp_path):
    stale = _task_plan()
    stale["global_plan"].append(
        {
            "phase_id": "cross_passage",
            "abstract_goal": "traverse the cross passage",
            "completion_evidence": "global view shows the second corner reached",
            "status": "active",
        }
    )
    stale["global_plan"][0]["status"] = "completed"
    stale["global_plan"][1]["status"] = "completed"
    stale["current_phase_id"] = "cross_passage"
    stale["current_phase_status"] = "active"

    corrected = json.loads(json.dumps(stale))
    corrected["global_plan"][1]["status"] = "active"
    corrected["global_plan"][2]["status"] = "pending"
    corrected["current_phase_id"] = "first_corner"

    client = FakeSharedQwenVL([stale, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = stale
    accepted_history = {
        "latest_checkpoint": "accepted_step_05",
        "steps": [
            {
                "checkpoint_after": "accepted_step_01",
                "method": "place_payload_on_arms",
            },
            {
                "checkpoint_after": "accepted_step_02",
                "method": "coordinated_linear_transport",
                "semantic_objective": "forward",
            },
            {
                "checkpoint_after": "accepted_step_03",
                "method": "coordinated_linear_transport",
                "semantic_objective": "forward",
            },
            {
                "checkpoint_after": "accepted_step_04",
                "method": "coordinated_alignment",
                "semantic_objective": "longitudinal",
            },
            {
                "checkpoint_after": "accepted_step_05",
                "method": "coordinated_linear_transport",
                "semantic_objective": "forward",
            },
        ],
    }

    result = pipeline.plan_task(
        {"accepted_route_action_history": accepted_history},
        artifact_id="attempt_10",
    )

    assert result["current_phase_id"] == "initial_straight"
    assert [phase["status"] for phase in result["global_plan"]] == [
        "active",
        "pending",
        "pending",
    ]
    assert pipeline.records[0]["valid"] is True
    assert pipeline.records[0]["host_schema_normalizations"] == [
        "global_plan statuses: aligned the persistent completion prefix to the "
        "trusted physical frontier 0 and restored one current phase"
    ]
    assert '"maximum_completed_prefix": 0' in client.calls[0]["user"]


def test_trusted_physical_frontier_prevents_false_route_phase_catch_up(tmp_path):
    previous = _task_plan()
    previous["global_plan"].extend(
        [
            {
                "phase_id": "cross_passage",
                "abstract_goal": "traverse the cross passage",
                "completion_evidence": "global view shows the second corner reached",
                "status": "completed",
            },
            {
                "phase_id": "second_corner",
                "abstract_goal": "negotiate the second corner",
                "completion_evidence": "global view shows final-passage alignment",
                "status": "active",
            },
        ]
    )
    previous["global_plan"][0]["status"] = "completed"
    previous["global_plan"][1]["status"] = "completed"
    previous["current_phase_id"] = "second_corner"
    previous["current_phase_status"] = "active"

    invalid = json.loads(json.dumps(previous))
    corrected = json.loads(json.dumps(previous))
    for index, phase in enumerate(corrected["global_plan"]):
        phase["status"] = "active" if index == 0 else "pending"
    corrected["current_phase_id"] = "initial_straight"
    corrected["current_phase_status"] = "active"

    client = FakeSharedQwenVL([invalid, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous
    result = pipeline.plan_task(
        {
            "accepted_route_action_history": {
                "steps": [
                    {
                        "checkpoint_after": "accepted_step_02",
                        "method": "coordinated_linear_transport",
                        "transition_outcome": "ACCEPT_PROGRESS",
                        "positive_route_memory_eligible": True,
                    },
                    {
                        "checkpoint_after": "accepted_step_03",
                        "method": "coordinated_pivot",
                        "transition_outcome": "ACCEPT_PROGRESS",
                        "positive_route_memory_eligible": True,
                    },
                    {
                        "checkpoint_after": "accepted_step_08",
                        "method": "coordinated_linear_transport",
                        "transition_outcome": "ACCEPT_PROGRESS",
                        "positive_route_memory_eligible": True,
                    },
                ]
            },
            "host_route_phase_guard": {
                "source": "trusted_checkpoint_physics",
                "current_topological_phase": "stage1_first_straight",
                "maximum_completed_prefix": 0,
                "coordinates_hidden_from_model": True,
            },
        },
        artifact_id="attempt_12",
    )

    assert result["current_phase_id"] == "initial_straight"
    assert [phase["status"] for phase in result["global_plan"]] == [
        "active",
        "pending",
        "pending",
        "pending",
    ]
    first_prompt = client.calls[0]["user"]
    assert '"action_evidence_maximum_prefix": 3' in first_prompt
    assert '"physical_maximum_completed_prefix": 0' in first_prompt
    assert "goal_xy" not in first_prompt


def test_accepted_action_label_cannot_override_physical_phase_regression(tmp_path):
    previous = _task_plan()
    previous["global_plan"][0]["status"] = "completed"
    previous["global_plan"][1]["status"] = "active"
    previous["current_phase_id"] = "first_corner"
    previous["current_phase_status"] = "active"

    regressed = json.loads(json.dumps(previous))
    regressed["global_plan"][0]["status"] = "active"
    regressed["global_plan"][1]["status"] = "pending"
    regressed["current_phase_id"] = "initial_straight"
    corrected = json.loads(json.dumps(previous))

    client = FakeSharedQwenVL([regressed, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous
    result = pipeline.plan_task(
        {
            "previous_subtask_outcome": "accepted",
            "accepted_route_action_history": {
                "latest_checkpoint": "accepted_step_04",
                "steps": [
                    {
                        "checkpoint_after": "accepted_step_02",
                        "method": "coordinated_linear_transport",
                        "semantic_objective": "forward",
                    },
                    {
                        "checkpoint_after": "accepted_step_04",
                        "method": "coordinated_linear_transport",
                        "semantic_objective": "backward",
                    },
                ],
            },
            "host_route_phase_guard": {
                "source": "trusted_checkpoint_physics",
                "current_topological_phase": "stage1_first_straight",
                "maximum_completed_prefix": 0,
                "coordinates_hidden_from_model": True,
            },
        },
        artifact_id="attempt_05",
    )

    assert result["current_phase_id"] == "initial_straight"
    assert [phase["status"] for phase in result["global_plan"]] == [
        "active",
        "pending",
    ]
    assert pipeline.records[0]["valid"] is True
    assert '"persistent_plan_floor_prefix": 0' in client.calls[0]["user"]


def test_rollback_phase_is_owned_by_restored_physical_frontier(tmp_path):
    previous = _task_plan()
    previous["global_plan"][0]["status"] = "completed"
    previous["global_plan"][1]["status"] = "active"
    previous["current_phase_id"] = "first_corner"
    previous["current_phase_status"] = "active"

    regressed = json.loads(json.dumps(previous))
    regressed["global_plan"][0]["status"] = "active"
    regressed["global_plan"][1]["status"] = "pending"
    regressed["current_phase_id"] = "initial_straight"
    corrected = json.loads(json.dumps(previous))

    client = FakeSharedQwenVL([regressed, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous
    result = pipeline.plan_task(
        {
            "previous_subtask_outcome": "rejected",
            "current_trusted_observation": {
                "runtime_state": {
                    "checkpoint": {"checkpoint_id": "accepted_step_04"}
                }
            },
            "accepted_route_action_history": {
                "latest_checkpoint": "accepted_step_04",
                "steps": [
                    {
                        "checkpoint_after": "accepted_step_02",
                        "method": "coordinated_linear_transport",
                        "semantic_objective": "forward",
                    },
                    {
                        "checkpoint_after": "accepted_step_04",
                        "method": "coordinated_linear_transport",
                        "semantic_objective": "backward",
                    },
                ],
            },
            "host_route_phase_guard": {
                "source": "trusted_checkpoint_physics",
                "current_topological_phase": "stage1_first_straight",
                "maximum_completed_prefix": 0,
                "coordinates_hidden_from_model": True,
            },
        },
        artifact_id="attempt_06",
    )

    assert result["current_phase_id"] == "initial_straight"
    assert [phase["status"] for phase in result["global_plan"]] == [
        "active",
        "pending",
    ]
    assert pipeline.records[0]["valid"] is True
    assert '"trusted_checkpoint_matches_history": true' in client.calls[0]["user"]


def test_phase_catch_up_cannot_skip_an_incomplete_phase(tmp_path):
    previous = _task_plan()
    previous["global_plan"].append(
        {
            "phase_id": "cross_passage",
            "abstract_goal": "traverse the cross passage",
            "completion_evidence": "global view shows the second corner reached",
            "status": "pending",
        }
    )
    skipped = json.loads(json.dumps(previous))
    skipped["global_plan"][0]["status"] = "completed"
    skipped["global_plan"][2]["status"] = "completed"
    skipped["current_phase_id"] = "first_corner"
    corrected = json.loads(json.dumps(previous))
    client = FakeSharedQwenVL([skipped, corrected])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = previous

    result = pipeline.plan_task(
        {
            "last_accepted_policy": {
                "robot_scripts": [
                    {
                        "robot_id": "front",
                        "code": (
                            "def policy(api):\n"
                            "    api.coordinated_linear_transport("
                            "direction='forward', distance_m=2.0, "
                            "speed_mps=0.5, ramp_s=1.5)\n"
                            "    api.stop_all()"
                        ),
                    }
                ]
            },
            "last_accepted_rollout": {"success": True},
        },
        artifact_id="attempt_03",
    )

    assert result["current_phase_id"] == "initial_straight"
    assert "contiguous prefix" in pipeline.records[0]["validation_error"]


def test_high_level_status_exposes_phase_action_roles_and_outcome(tmp_path):
    pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([]),
        output_dir=tmp_path,
    )

    ready = pipeline.publish_high_level_status(
        "policy_ready",
        attempt=12,
        checkpoint_id="accepted_step_03",
        task_plan=_task_plan(),
        policy=_model_policy_payload(),
        next_system_operation="run preflight",
    )
    rejected = pipeline.publish_high_level_status(
        "rejected",
        attempt=12,
        checkpoint_id="accepted_step_03",
        last_outcome={
            "status": "rejected",
            "failure_type": "route_regression",
            "rollback_performed": True,
        },
    )

    assert ready["current_phase"]["phase_id"] == "initial_straight"
    assert ready["current_phase"]["index"] == 1
    assert ready["current_subtask"]["subtask_id"] == "initial_straight_segment"
    assert ready["next_action"] == {
        "primitive": "coordinated_linear_transport",
        "semantic_objective": "forward",
        "formation_strategy": "side-by-side loaded transport",
        "coordination_mode": "sync",
        "synchronization_intent": "one identical atomic transaction",
    }
    assert ready["team_assignment"]["front"]["role"] == "co-carrier"
    assert rejected["event_sequence"] == 2
    assert rejected["last_outcome"]["failure_type"] == "route_regression"
    latest = json.loads(
        (
            tmp_path / "model_roles" / "live_high_level_status.json"
        ).read_text(encoding="utf-8")
    )
    history = [
        json.loads(line)
        for line in (
            tmp_path / "model_roles" / "high_level_status_history.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert latest == rejected
    assert [item["runtime_status"] for item in history] == [
        "policy_ready",
        "rejected",
    ]
    assert not list((tmp_path / "model_roles").glob(".*.tmp"))


def test_high_level_status_sequence_continues_when_run_is_resumed(tmp_path):
    first = ModelRolePipeline(
        client=FakeSharedQwenVL([]),
        output_dir=tmp_path,
    )
    first.publish_high_level_status(
        "accepted",
        attempt=3,
        checkpoint_id="accepted_step_03",
        task_plan=_task_plan(),
    )

    resumed = ModelRolePipeline(
        client=FakeSharedQwenVL([]),
        output_dir=tmp_path,
    )
    status = resumed.publish_high_level_status(
        "planning",
        attempt=4,
        checkpoint_id="accepted_step_03",
    )

    assert status["event_sequence"] == 2
    assert status["current_phase"]["phase_id"] == "initial_straight"
    assert len(
        (
            tmp_path / "model_roles" / "high_level_status_history.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ) == 2


def test_evolution_manager_retries_invalid_mutation_level(tmp_path):
    invalid = {**_manager(), "mutation_level": "rewrite_everything"}
    client = FakeSharedQwenVL([invalid, _manager("strategy_mutation")])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path, response_retries=1)

    result = pipeline.decide_mutation({}, artifact_id="attempt_02")

    assert result["mutation_level"] == "strategy_mutation"
    records = pipeline.records
    assert records[0]["valid"] is False
    assert "mutation_level" in records[0]["validation_error"]
    assert records[1]["valid"] is True


def test_vlm_analyzer_retry_requires_nested_failure_analysis(tmp_path):
    valid = _failure_analysis()
    invalid = {
        key: value
        for key, value in valid.items()
        if key != "failure_analysis"
    }
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"test")
    client = FakeSharedQwenVL([invalid, valid])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    result = pipeline.analyze_rollout(
        {},
        artifact_id="initial_observation",
        image_path=frame,
    )

    assert result["failure_analysis"]["coordination_defect"] is True
    retry_prompt = client.calls[1]["user"]
    assert "required top-level object" in retry_prompt
    assert "legacy visual-only schema" in retry_prompt


def test_vlm_analyzer_rejects_string_route_risk_and_retries(tmp_path):
    invalid = _failure_analysis()
    invalid["immediate_route_direction_risk"] = "High"
    invalid["team_heading_relative_to_visible_free_space"] = "Toward passage"
    invalid["team_payload_posture_relative_to_visible_passage"] = (
        "Tilted sideways"
    )
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"test")
    client = FakeSharedQwenVL([invalid, _failure_analysis()])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    result = pipeline.analyze_rollout(
        {},
        artifact_id="attempt_05",
        image_path=frame,
    )

    assert result["immediate_route_direction_risk"] is False
    assert pipeline.records[0]["valid"] is False
    assert "exact JSON boolean" in pipeline.records[0]["validation_error"]
    assert "team_heading_relative_to_visible_free_space" in pipeline.records[0][
        "validation_error"
    ]
    assert "team_payload_posture_relative_to_visible_passage" in pipeline.records[
        0
    ]["validation_error"]
    assert "literal JSON true" in client.calls[1]["user"]
    assert "High" in client.calls[1]["user"]


def test_vlm_analyzer_rejects_planning_label_as_top_level_failure(tmp_path):
    invalid = _failure_analysis()
    invalid["failure_type"] = "strategy_mismatch"
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"test")
    client = FakeSharedQwenVL([invalid, _failure_analysis()])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    result = pipeline.analyze_rollout(
        {},
        artifact_id="attempt_05",
        image_path=frame,
    )

    assert result["failure_type"] == "none"
    assert pipeline.records[0]["valid"] is False
    assert "failure_type is safety-only" in pipeline.records[0][
        "validation_error"
    ]
    assert "failure_analysis.primary_failure" in client.calls[1]["user"]


def test_mutation_retry_explains_child_primitive_mode_contract(tmp_path):
    invalid = _pivot_strategy_mutation()
    invalid["child_strategy"]["coordination_mode"] = "sync"
    invalid["child_strategy"]["formation_strategy"] = "sync pivot"
    invalid["child_strategy"]["team_assignment"] = {
        robot_id: {
            "role": "payload_stabilizer",
            "objective": "preserve synchronized support",
        }
        for robot_id in ("front", "back")
    }
    client = FakeSharedQwenVL([invalid, _pivot_strategy_mutation()])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    def validate(candidate: dict) -> None:
        _validate_plan_guided_mutation(
            candidate,
            manager_decision={"mutation_level": "strategy_mutation"},
            parent_strategy=_task_plan()["current_subtask"]["strategy"],
            allowed_sdk_methods={
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
        )

    result = pipeline.mutate_strategy(
        {},
        artifact_id="attempt_11",
        validator=validate,
    )

    assert result["child_strategy"]["coordination_mode"] == "async"
    retry_prompt = client.calls[1]["user"]
    assert "hard SDK contract requires coordination_mode='async'" in retry_prompt
    assert "exactly one robot role=inner" in retry_prompt


def test_mutation_retry_uses_exact_child_semantic_objective(tmp_path):
    invalid = _pivot_strategy_mutation()
    invalid["child_strategy"]["semantic_objective"] = "align_and_advance"
    client = FakeSharedQwenVL([invalid, _pivot_strategy_mutation()])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    def validate(candidate: dict) -> None:
        _validate_plan_guided_mutation(
            candidate,
            manager_decision={"mutation_level": "strategy_mutation"},
            parent_strategy=_task_plan()["current_subtask"]["strategy"],
            allowed_sdk_methods={
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
        )

    result = pipeline.mutate_strategy(
        {
            "orchestrator_proposed_strategy": (
                _pivot_strategy_mutation()["child_strategy"]
            )
        },
        artifact_id="attempt_11",
        validator=validate,
    )

    assert result["child_strategy"]["semantic_objective"] == "counterclockwise"
    retry_prompt = client.calls[1]["user"]
    assert "exactly one unchanged token" in retry_prompt
    assert "Do not use descriptive prose such as align_and_advance" in retry_prompt
    assert "Keep the latest child_strategy.primitive unchanged" in retry_prompt
    assert "different child primitive on an earlier retry is obsolete" in retry_prompt


def test_formal_strategy_mutation_does_not_tabu_exhausted_linear_family():
    parent = _pivot_strategy_mutation()["child_strategy"]
    child = {
        **_task_plan()["current_subtask"]["strategy"],
        "semantic_objective": "backward",
        "team_assignment": {
            robot_id: {
                "role": robot_id,
                "objective": "move backward while preserving support",
            }
            for robot_id in ("front", "back")
        },
    }
    mutation = {
        "mutation_level": "strategy_mutation",
        "reason": "replace the rejected pivot with passage translation",
        "parent_strategy": parent,
        "child_strategy": child,
        "changed_dimensions": [
            "maneuver primitive",
            "semantic objective",
            "coordination mode",
            "formation strategy",
        ],
        "inheritance": "preserve mission, checkpoint progress, and support",
    }
    rejected_tiers = {
        _loaded_linear_team_transaction_signature(
            direction="backward",
            distance_m=distance_m,
        )
        for distance_m in (0.25, 0.5, 0.75, 1.0)
    }
    clearance = {
        "translation_candidates": {
            "backward": {"relation": "open", "clearance_m": 1.291},
        }
    }

    _validate_plan_guided_mutation(
        mutation,
        manager_decision={"mutation_level": "strategy_mutation"},
        parent_strategy=parent,
        allowed_sdk_methods={
            "coordinated_linear_transport",
            "coordinated_pivot",
        },
        host_wall_clearance=clearance,
        forbidden_transaction_signatures=rejected_tiers,
        model_owns_route=True,
    )
    with pytest.raises(
        ModelRoleError,
        match="no untried model-supported distance tier",
    ):
        _validate_plan_guided_mutation(
            mutation,
            manager_decision={"mutation_level": "strategy_mutation"},
            parent_strategy=parent,
            allowed_sdk_methods={
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
            host_wall_clearance=clearance,
            forbidden_transaction_signatures=rejected_tiers,
            model_owns_route=False,
        )

    clearance["translation_candidates"]["backward"]["clearance_m"] = 1.6
    _validate_plan_guided_mutation(
        mutation,
        manager_decision={"mutation_level": "strategy_mutation"},
        parent_strategy=parent,
        allowed_sdk_methods={
            "coordinated_linear_transport",
            "coordinated_pivot",
        },
        host_wall_clearance=clearance,
        forbidden_transaction_signatures={
            _loaded_linear_team_transaction_signature(
                direction="backward",
                distance_m=1.0,
            )
        },
        model_owns_route=True,
    )


def test_mutation_pipeline_types_infeasible_child_cycle_outcome(tmp_path):
    proposal = _pivot_strategy_mutation()
    pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([proposal, proposal]),
        output_dir=tmp_path,
        response_retries=1,
    )

    def reject_candidate(_candidate: dict) -> None:
        raise ModelRoleError("no executable child at this selected level")

    with pytest.raises(ModelRoleStallError) as caught:
        pipeline.mutate_strategy(
            {"evolution_manager": {"mutation_level": "strategy_mutation"}},
            artifact_id="attempt_21",
            validator=reject_candidate,
        )

    assert caught.value.cycle_outcome == NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL
    assert caught.value.selected_mutation_level == "strategy_mutation"
    assert caught.value.validation_error_kinds == (
        "candidate_feasibility",
        "candidate_feasibility",
    )


def test_strict_strategy_mutation_retries_when_linear_tiers_are_exhausted(
    tmp_path,
):
    parent = _pivot_strategy_mutation()["child_strategy"]

    def mutation(direction: str) -> dict:
        return {
            "mutation_level": "strategy_mutation",
            "reason": "replace the rejected pivot with passage translation",
            "parent_strategy": parent,
            "child_strategy": {
                **_task_plan()["current_subtask"]["strategy"],
                "semantic_objective": direction,
                "team_assignment": {
                    robot_id: {
                        "role": robot_id,
                        "objective": f"move {direction} while preserving support",
                    }
                    for robot_id in ("front", "back")
                },
            },
            "changed_dimensions": [
                "maneuver primitive",
                "semantic objective",
                "coordination mode",
                "formation strategy",
            ],
            "inheritance": "preserve mission, checkpoint progress, and support",
        }

    client = FakeSharedQwenVL([mutation("backward"), mutation("left")])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )
    clearance = {
        "translation_candidates": {
            "backward": {"relation": "open", "clearance_m": 1.291},
            "left": {"relation": "open", "clearance_m": 1.2},
        }
    }
    rejected_backward_tiers = {
        _loaded_linear_team_transaction_signature(
            direction="backward",
            distance_m=distance_m,
        )
        for distance_m in (0.25, 0.5, 0.75, 1.0)
    }

    def validate(candidate: dict) -> None:
        _validate_plan_guided_mutation(
            candidate,
            manager_decision={"mutation_level": "strategy_mutation"},
            parent_strategy=parent,
            allowed_sdk_methods={
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
            host_wall_clearance=clearance,
            forbidden_transaction_signatures=rejected_backward_tiers,
            model_owns_route=False,
        )

    result = pipeline.mutate_strategy(
        {},
        artifact_id="attempt_39",
        validator=validate,
    )

    assert result["child_strategy"]["semantic_objective"] == "left"
    assert pipeline.records[0]["valid"] is False
    retry_prompt = client.calls[1]["user"]
    assert "no legal Coder parameter tier" in retry_prompt
    assert "Change child_strategy primitive or semantic_objective" in retry_prompt


def test_plan_guided_coder_receives_rejected_executable_transactions(tmp_path):
    coder_payload = _sync_linear_team_policy_payload()
    coder_payload["verification_contract"] = {}
    client = FakeSharedQwenVL([coder_payload])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    task_plan = _task_plan()
    strategy = {
        **task_plan["current_subtask"]["strategy"],
        "team_assignment": task_plan["current_subtask"]["team_assignment"],
    }
    rejected_transaction = (
        "robot=front coordination_mode=sync\n"
        "api.coordinated_linear_transport(direction='forward', distance_m=1.5, "
        "speed_mps=0.5, ramp_s=1.5)"
    )

    _parsed, policy = _generate_plan_guided_team_policy(
        role_pipeline=pipeline,
        task_plan=task_plan,
        execution_strategy=strategy,
        generation_context={
            "recent_rejected_team_transactions": ["sha256:test"],
            "recent_rejected_executable_transactions": [rejected_transaction],
            "host_authored_wall_clearance": {
                "translation_candidates": {
                    "forward": {"relation": "open", "clearance_m": 2.0},
                }
            },
        },
        attempt=2,
        fallback_id="attempt_02",
        parent_script_id="attempt_01",
        require_team_plan=True,
        expected_method="",
        forbidden_transaction_signatures=(),
        current_policy=None,
        repair_level="",
        repair_failure_type="",
    )

    assert "forbidden_executable_transactions" in client.calls[0]["user"]
    assert "distance_m=1.5" in client.calls[0]["user"]
    assert "direction='forward'" in client.calls[0]["user"]
    committed = policy.verification_contract["committed_subtask"]
    assert committed["immutable_by_coder"] is True
    assert committed["subtask_id"] == "initial_straight_segment"
    assert committed["primitive"] == "coordinated_linear_transport"


def test_plan_guided_coder_preserves_turn_clearance_setup_contract(tmp_path):
    payload = _sync_linear_team_policy_payload()
    for script in payload["robot_scripts"].values():
        script["code"] = script["code"].replace(
            "direction='forward'", "direction='backward'"
        )
    client = FakeSharedQwenVL([payload])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    task_plan = _task_plan()
    task_plan["current_subtask"]["strategy"]["semantic_objective"] = "backward"
    strategy = {
        **task_plan["current_subtask"]["strategy"],
        "team_assignment": task_plan["current_subtask"]["team_assignment"],
    }
    state = {
        "checkpoint": {"checkpoint_id": "accepted_step_18"},
        "turn_episode": {
            "active": True,
            "status": "turning",
            "purpose": "local_navigation_turn",
            "reference": "final_target",
            "direction": "counterclockwise",
        },
    }
    clearance = {
        "available": True,
        "translation_candidates": {
            "backward": {"relation": "open", "clearance_m": 1.6},
        },
        "open_translation_directions": ["backward"],
        "turn_candidates": {
            "counterclockwise": {
                "relation": "blocked",
                "max_safe_angle_deg": 0.0,
            },
        },
    }
    contract = {
        "active": True,
        "scope_checkpoint_id": "accepted_step_18",
        "allowed_translation_directions": ["backward"],
        "required_primitive": "coordinated_linear_transport",
        "required_coordination_mode": "sync",
        "setup_only": True,
    }
    context = {
        "observation": {"runtime_state": state},
        "host_authored_wall_clearance": clearance,
        "turn_clearance_replan_contract": contract,
    }

    _parsed, policy = _generate_plan_guided_team_policy(
        role_pipeline=pipeline,
        task_plan=task_plan,
        execution_strategy=strategy,
        generation_context=context,
        attempt=20,
        fallback_id="attempt_20",
        parent_script_id="attempt_19",
        require_team_plan=True,
        expected_method="",
        forbidden_transaction_signatures=(),
        current_policy=None,
        repair_level="",
        repair_failure_type="",
    )

    assert policy.verification_contract["turn_clearance_replan"] == contract

    missing_contract_pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([payload]),
        output_dir=tmp_path / "missing_contract",
    )
    with pytest.raises(ModelAgentError, match="system contract propagation error"):
        _generate_plan_guided_team_policy(
            role_pipeline=missing_contract_pipeline,
            task_plan=task_plan,
            execution_strategy=strategy,
            generation_context={
                "observation": {"runtime_state": state},
                "host_authored_wall_clearance": clearance,
            },
            attempt=20,
            fallback_id="attempt_20",
            parent_script_id="attempt_19",
            require_team_plan=True,
            expected_method="",
            forbidden_transaction_signatures=(),
            current_policy=None,
            repair_level="",
            repair_failure_type="",
        )


def test_formal_coder_binds_advisory_clearance_setup_as_route_recovery(tmp_path):
    payload = _sync_linear_team_policy_payload()
    for script in payload["robot_scripts"].values():
        script["code"] = script["code"].replace(
            "direction='forward'", "direction='backward'"
        )
    client = FakeSharedQwenVL([payload])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    task_plan = _task_plan()
    task_plan["current_subtask"]["strategy"]["semantic_objective"] = "backward"
    task_plan["current_subtask"]["strategy"]["synchronization_intent"] = (
        "translate_backward_to_gain_clearance"
    )
    strategy = {
        **task_plan["current_subtask"]["strategy"],
        "team_assignment": task_plan["current_subtask"]["team_assignment"],
    }
    advisory_clearance = {
        "active": True,
        "source": "active_route_plus_authored_wall_geometry",
        "scope_checkpoint_id": "accepted_step_08",
        "route_aligned_translation_directions": ["forward"],
        "clearance_setup_translation_directions": ["backward", "left", "right"],
        "setup_only": True,
        "phase_completion_authorized": False,
    }

    _parsed, policy = _generate_plan_guided_team_policy(
        role_pipeline=pipeline,
        task_plan=task_plan,
        execution_strategy=strategy,
        generation_context={
            "formal_v50_lite": True,
            "advisory_route_clearance": advisory_clearance,
            "current_checkpoint_id": "accepted_step_08",
        },
        attempt=14,
        fallback_id="attempt_14",
        parent_script_id="attempt_13",
        require_team_plan=True,
        expected_method="",
        forbidden_transaction_signatures=(),
        current_policy=None,
        repair_level="",
        repair_failure_type="",
    )

    recovery = policy.verification_contract["route_recovery"]
    assert recovery["active"] is True
    assert recovery["setup_only"] is True
    assert recovery["phase_completion_authorized"] is False
    assert recovery["positive_route_memory_eligible"] is False
    assert recovery["semantic_objective"] == "backward"
    assert "route_clearance_conflict" not in policy.verification_contract

    bound = _committed_route_recovery_contract(
        execution_strategy=strategy,
        generation_context={
            "formal_v50_lite": True,
            "advisory_route_clearance": advisory_clearance,
            "current_checkpoint_id": "accepted_step_08",
        },
        committed_subtask=task_plan["current_subtask"],
    )
    assert bound["semantic_objective"] == "backward"


def _linear_recovery_strategy(
    *,
    objective: str = "backward",
    mode: str = "sync",
) -> dict:
    return {
        "primitive": "coordinated_linear_transport",
        "semantic_objective": objective,
        "coordination_mode": mode,
    }


def _route_clearance_setup_contract(
    *,
    scope_checkpoint_id: str = "accepted_step_08",
    source: str = "host_route_clearance",
) -> dict:
    return {
        "active": True,
        "source": source,
        "scope_checkpoint_id": scope_checkpoint_id,
        "route_aligned_translation_directions": ["forward"],
        "clearance_setup_translation_directions": ["backward"],
        "setup_only": True,
        "phase_completion_authorized": False,
        "rejected_motion": {"semantic_objective": "forward"},
    }


@pytest.mark.parametrize(
    "generation_context",
    [
        {"current_checkpoint_id": "accepted_step_08"},
        {
            "current_checkpoint_id": "accepted_step_08",
            "advisory_route_clearance": {
                key: value
                for key, value in _route_clearance_setup_contract().items()
                if key != "scope_checkpoint_id"
            },
        },
        {
            "current_checkpoint_id": "accepted_step_08",
            "advisory_route_clearance": _route_clearance_setup_contract(
                scope_checkpoint_id="accepted_step_07"
            ),
        },
    ],
    ids=("no_host_contract", "missing_scope", "stale_scope"),
)
def test_committed_route_recovery_requires_checkpoint_scoped_host_authority(
    generation_context,
):
    assert (
        _committed_route_recovery_contract(
            execution_strategy=_linear_recovery_strategy(),
            generation_context=generation_context,
            committed_subtask={"setup_only": True},
        )
        == {}
    )


def test_committed_route_recovery_accepts_matching_route_clearance_authority():
    contract = _route_clearance_setup_contract()

    recovery = _committed_route_recovery_contract(
        execution_strategy=_linear_recovery_strategy(),
        generation_context={
            "current_checkpoint_id": "accepted_step_08",
            "advisory_route_clearance": contract,
        },
        committed_subtask={"setup_only": True},
    )

    assert recovery["source"] == "host_route_clearance"
    assert recovery["host_authoritative"] is True
    assert recovery["scope_checkpoint_id"] == "accepted_step_08"
    assert recovery["semantic_objective"] == "backward"
    assert recovery["rejected_motion"] == {"semantic_objective": "forward"}


def test_committed_route_recovery_does_not_mix_stale_recovery_with_live_clearance():
    live_clearance = _route_clearance_setup_contract(
        source="live_route_clearance"
    )
    stale_recovery = {
        "active": True,
        "source": "stale_route_recovery",
        "scope_checkpoint_id": "accepted_step_07",
        "rejected_motion": {"semantic_objective": "left"},
    }

    recovery = _committed_route_recovery_contract(
        execution_strategy=_linear_recovery_strategy(),
        generation_context={
            "current_checkpoint_id": "accepted_step_08",
            "route_recovery_contract": stale_recovery,
            "advisory_route_clearance": live_clearance,
        },
        committed_subtask={"setup_only": True},
    )

    assert recovery["source"] == "live_route_clearance"
    assert recovery["scope_checkpoint_id"] == "accepted_step_08"
    assert recovery["rejected_motion"] == {"semantic_objective": "forward"}


@pytest.mark.parametrize(
    ("strategy", "expected_authorized"),
    [
        (_linear_recovery_strategy(), True),
        (_linear_recovery_strategy(objective="forward"), False),
        (_linear_recovery_strategy(mode="async"), False),
        (
            {
                "primitive": "coordinated_pivot",
                "semantic_objective": "clockwise",
                "coordination_mode": "async",
            },
            False,
        ),
    ],
    ids=("exact_match", "wrong_objective", "wrong_mode", "wrong_primitive"),
)
def test_turn_clearance_recovery_requires_exact_committed_strategy_match(
    strategy,
    expected_authorized,
):
    contract = {
        "active": True,
        "source": "host_turn_clearance",
        "scope_checkpoint_id": "accepted_step_18",
        "allowed_translation_directions": ["backward"],
        "required_primitive": "coordinated_linear_transport",
        "required_coordination_mode": "sync",
        "setup_only": True,
        "phase_completion_authorized": False,
    }

    recovery = _committed_route_recovery_contract(
        execution_strategy=strategy,
        generation_context={
            "current_checkpoint_id": "accepted_step_18",
            "turn_clearance_replan_contract": contract,
        },
        committed_subtask={"setup_only": True},
    )

    assert bool(recovery) is expected_authorized


def test_committed_route_recovery_requires_a_trusted_current_checkpoint():
    assert (
        _committed_route_recovery_contract(
            execution_strategy=_linear_recovery_strategy(),
            generation_context={
                "advisory_route_clearance": _route_clearance_setup_contract()
            },
            committed_subtask={"setup_only": True},
        )
        == {}
    )


def test_strategy_mutation_replan_contract_strips_role_objective_prose():
    mutation = _pivot_strategy_mutation()
    mutation["child_strategy"]["synchronization_intent"] = (
        "maintain support while correcting formation"
    )
    mutation["child_strategy"]["team_assignment"]["back"]["objective"] = (
        "move body-forward to reduce longitudinal stagger"
    )
    contract = _strategy_mutation_replan_contract(
        mutation["child_strategy"]
    )

    assert contract["primitive"] == "coordinated_pivot"
    assert contract["semantic_objective"] == "counterclockwise"
    assert contract["team_assignment"] == {
        "front": {"role": "inner"},
        "back": {"role": "outer"},
    }
    assert "objective" not in contract["team_assignment"]["back"]


def test_mutation_preserves_nonexecutable_parameter_prose_in_inheritance(tmp_path):
    response = _pivot_strategy_mutation()
    response["inheritance"] = [
        (
            "api.coordinated_pivot(turn_direction='counterclockwise', "
            "angle_deg=30.0, duration_s=8.0)"
        ),
        "api.stop_all()",
    ]
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.mutate_strategy({}, artifact_id="attempt_11")

    assert result["inheritance"] == response["inheritance"]
    assert len(client.calls) == 1


def test_task_orchestrator_audits_scalar_list_normalization(tmp_path):
    response = _task_plan()
    response["constraints"] = "preserve dual payload support"
    response["current_subtask"]["preserve"] = "accepted route progress"
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.plan_task({}, artifact_id="attempt_01")

    assert result["constraints"] == ["preserve dual payload support"]
    assert result["current_subtask"]["preserve"] == ["accepted route progress"]
    assert pipeline.records[0]["host_schema_normalizations"] == [
        "current_subtask.preserve: string -> singleton list",
        "constraints: string -> singleton list",
    ]


def test_task_orchestrator_lifts_nested_team_assignment(tmp_path):
    response = _task_plan()
    assignment = response["current_subtask"].pop("team_assignment")
    response["current_subtask"]["strategy"]["team_assignment"] = assignment
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.plan_task({}, artifact_id="attempt_02")

    assert result["current_subtask"]["team_assignment"] == assignment
    assert pipeline.records[0]["host_schema_normalizations"] == [
        "current_subtask.team_assignment: copied from strategy"
    ]


def test_task_orchestrator_retry_demands_exact_semantic_objective_token(tmp_path):
    invalid = _task_plan()
    invalid["current_subtask"]["strategy"] = {
        **invalid["current_subtask"]["strategy"],
        "primitive": "coordinated_alignment",
        "coordination_mode": "async",
        "semantic_objective": "align support center to target",
    }
    corrected = json.loads(json.dumps(invalid))
    corrected["current_subtask"]["strategy"]["semantic_objective"] = "lateral"
    client = FakeSharedQwenVL([invalid, corrected])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    def validate(plan):
        objective = plan["current_subtask"]["strategy"]["semantic_objective"]
        if objective != "lateral":
            raise ModelRoleError(
                "semantic_objective for api.coordinated_alignment must be one "
                "of ['lateral', 'longitudinal', 'yaw']; got "
                f"{objective!r}"
            )

    result = pipeline.plan_task(
        {"exact_sdk_primitive_names": ["coordinated_alignment"]},
        artifact_id="attempt_04",
        validator=validate,
    )

    assert result["current_subtask"]["strategy"]["semantic_objective"] == "lateral"
    assert "exactly one token" in client.calls[1]["user"]
    assert "Do not paraphrase" in client.calls[1]["user"]


def test_task_orchestrator_normalizes_single_mode_primitive_contract(tmp_path):
    response = _task_plan()
    response["current_subtask"]["strategy"] = {
        **response["current_subtask"]["strategy"],
        "primitive": "coordinated_alignment",
        "coordination_mode": "sync",
        "semantic_objective": "longitudinal",
    }
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.plan_task({}, artifact_id="attempt_04")

    assert result["current_subtask"]["strategy"]["coordination_mode"] == "async"
    assert pipeline.records[0]["host_schema_normalizations"] == [
        "current_subtask.strategy.coordination_mode: sync -> async "
        "(api.coordinated_alignment contract)"
    ]


def test_sync_team_policy_uses_one_shared_coder_call(tmp_path):
    client = FakeSharedQwenVL([_model_policy_payload()])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    policy = pipeline.generate_team_policy(
        {"current_subtask": _task_plan()["current_subtask"]},
        artifact_id="attempt_06",
        coordination_mode="sync",
    )

    assert policy["script_id"] == "attempt_01"
    assert [record["role"] for record in pipeline.records] == ["coder"]
    audit = json.loads(
        (
            tmp_path
            / "model_roles"
            / "team_policy_generation_attempt_06.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["logical_model_calls"] == ["coder:sync_team"]


def test_external_policy_normalization_is_recorded_in_role_audit(tmp_path):
    client = FakeSharedQwenVL([_model_policy_payload()])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def normalize(payload):
        payload["host_schema_normalizations"] = [
            "coordinated_alignment.delta: preserved model sign and magnitude"
        ]

    pipeline.code_policy(
        {},
        artifact_id="attempt_06",
        validator=normalize,
    )

    assert pipeline.records[0]["host_schema_normalizations"] == [
        "coordinated_alignment.delta: preserved model sign and magnitude"
    ]


def test_async_team_policy_uses_two_coders_and_one_team_review(tmp_path):
    client = FakeSharedQwenVL(
        [
            _async_robot_draft("front"),
            _async_robot_draft("back"),
            _async_team_review_payload(),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    policy = pipeline.generate_team_policy(
        {
            "current_subtask": _task_plan()["current_subtask"],
            "evolution_manager": {"mutation_level": "strategy_mutation"},
            "current_policy": {"marker": "OLD_POLICY_MUST_NOT_REACH_REVIEW"},
            "evaluation": {"marker": "OLD_EVALUATION_MUST_NOT_REACH_REVIEW"},
            "verified_action_memory": [
                {"marker": "OLD_MEMORY_MUST_NOT_REACH_REVIEW"}
            ],
        },
        artifact_id="attempt_07",
        coordination_mode="async",
    )

    assert policy["script_id"] == "async_team_transaction"
    assert policy["team_review"]["decision"] == "approve_drafts"
    assert [record["role"] for record in pipeline.records] == [
        "coder",
        "coder",
        "code_repair",
    ]
    audit = json.loads(
        (
            tmp_path
            / "model_roles"
            / "team_policy_generation_attempt_07.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["logical_model_calls"] == [
        "coder:front",
        "coder:back",
        "code_repair:team_integration",
    ]
    review_prompt = client.calls[2]["user"]
    assert "front_draft" in review_prompt
    assert "back_draft" in review_prompt
    assert '"mutation_level": "strategy_mutation"' in review_prompt
    assert "OLD_POLICY_MUST_NOT_REACH_REVIEW" not in review_prompt
    assert "OLD_EVALUATION_MUST_NOT_REACH_REVIEW" not in review_prompt
    assert "OLD_MEMORY_MUST_NOT_REACH_REVIEW" not in review_prompt


def test_invalid_team_repair_cannot_replace_valid_coder_drafts(tmp_path):
    invalid_code = (
        "def policy(api):\n"
        "    api.coordinated_alignment("
        "axis='longitudinal', delta=9.0, duration_s=2.0)\n"
        "    api.stop_all()\n"
    )
    invalid_review = {
        "decision": "repair_drafts",
        "conflicts": ["invented conflict"],
        "repaired_code": {"front": invalid_code, "back": invalid_code},
        "rationale": "replace both valid drafts",
    }
    client = FakeSharedQwenVL(
        [
            _async_robot_draft("front"),
            _async_robot_draft("back"),
            *([invalid_review] * 5),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def reject_invalid_repair(candidate):
        assert "robot_scripts" in candidate
        if any(
            "delta=9.0" in script["code"]
            for script in candidate["robot_scripts"].values()
        ):
            raise ModelRoleError("exact verifier-rejected executable AST")

    policy = pipeline.generate_team_policy(
        {"task_orchestrator_plan": _task_plan()},
        artifact_id="attempt_07",
        coordination_mode="async",
        validator=reject_invalid_repair,
    )

    assert policy["team_review"]["source"] == "host_validated_draft_fallback"
    assert all(
        "delta=9.0" not in script["code"]
        for script in policy["robot_scripts"].values()
    )


def test_async_joint_model_agent_conflict_still_reaches_code_repair(tmp_path):
    """Production validators raise ModelAgentError; repair must still run."""

    front = _pivot_robot_draft("front")
    back = _pivot_robot_draft("back")
    # Match the V58 GPT seed_000 conflict: shared angle must match, roles differ.
    front["role"] = "outer"
    back["role"] = "inner"
    front["code"] = (
        "def policy(api):\n"
        "    api.coordinated_pivot("
        "turn_direction='counterclockwise', angle_deg=10.0, "
        "travel_direction='forward', duration_s=8.0, role='outer', "
        "swing_radius_m=0.8)\n"
        "    api.stop_all()\n"
    )
    back["code"] = (
        "def policy(api):\n"
        "    api.coordinated_pivot("
        "turn_direction='counterclockwise', angle_deg=20.0, "
        "travel_direction='forward', duration_s=8.0, role='inner', "
        "swing_radius_m=0.8)\n"
        "    api.stop_all()\n"
    )
    repaired_front = front["code"].replace("angle_deg=10.0", "angle_deg=15.0")
    repaired_back = back["code"].replace("angle_deg=20.0", "angle_deg=15.0")
    client = FakeSharedQwenVL(
        [
            front,
            back,
            {
                "decision": "repair_drafts",
                "conflicts": ["front/back angle_deg mismatch"],
                "repaired_code": {
                    "front": repaired_front,
                    "back": repaired_back,
                },
                "rationale": "unify shared pivot angle",
            },
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    seen_angles: list[tuple[float, float]] = []

    def production_like_validator(candidate):
        scripts = candidate.get("robot_scripts", {})
        angles = []
        for robot_id in ("front", "back"):
            code = str(scripts[robot_id]["code"])
            match = re.search(r"angle_deg=([0-9]+(?:\.[0-9]+)?)", code)
            assert match is not None
            angles.append(float(match.group(1)))
        seen_angles.append((angles[0], angles[1]))
        if angles[0] != angles[1]:
            raise ModelAgentError(
                "model team transaction failed validation: "
                "coordination primitive requires matching front/back "
                f"angle_deg; got {{'front': {angles[0]}, 'back': {angles[1]}}}"
            )

    policy = pipeline.generate_team_policy(
        {
            "task_orchestrator_plan": _pivot_task_plan(),
            "committed_execution_strategy": {
                "primitive": "coordinated_pivot",
                "coordination_mode": "async",
            },
        },
        artifact_id="attempt_07",
        coordination_mode="async",
        validator=production_like_validator,
    )

    assert [record["role"] for record in pipeline.records] == [
        "coder",
        "coder",
        "code_repair",
    ]
    assert seen_angles[0] == (10.0, 20.0)
    assert seen_angles[-1] == (15.0, 15.0)
    assert policy["team_review"]["decision"] == "repair_drafts"
    assert "angle_deg=15.0" in policy["robot_scripts"]["front"]["code"]
    assert "angle_deg=15.0" in policy["robot_scripts"]["back"]["code"]


def test_async_prevalidation_does_not_accept_named_runtime_error(tmp_path):
    """A same-named foreign exception is a Host error, not a draft conflict."""

    class ModelAgentError(RuntimeError):
        pass

    client = FakeSharedQwenVL(
        [_async_robot_draft("front"), _async_robot_draft("back")]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def broken_host_validator(candidate):
        assert "robot_scripts" in candidate
        raise ModelAgentError("foreign host validator bug")

    with pytest.raises(ModelAgentError, match="foreign host validator bug"):
        pipeline.generate_team_policy(
            {"task_orchestrator_plan": _task_plan()},
            artifact_id="attempt_07",
            coordination_mode="async",
            validator=broken_host_validator,
        )

    assert len(client.calls) == 2
    assert [record["role"] for record in pipeline.records] == ["coder", "coder"]


def test_async_prevalidation_propagates_provider_fatal(tmp_path):
    client = FakeSharedQwenVL(
        [_async_robot_draft("front"), _async_robot_draft("back")]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def unavailable_host_validator(candidate):
        assert "robot_scripts" in candidate
        raise ProviderFatalError("validator dependency unavailable", http_status=503)

    with pytest.raises(ProviderFatalError, match="validator dependency unavailable"):
        pipeline.generate_team_policy(
            {"task_orchestrator_plan": _task_plan()},
            artifact_id="attempt_07",
            coordination_mode="async",
            validator=unavailable_host_validator,
        )

    assert len(client.calls) == 2
    assert [record["role"] for record in pipeline.records] == ["coder", "coder"]


def test_async_prevalidation_propagates_unexpected_host_runtime_error(tmp_path):
    client = FakeSharedQwenVL(
        [_async_robot_draft("front"), _async_robot_draft("back")]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def broken_host_validator(candidate):
        assert "robot_scripts" in candidate
        raise RuntimeError("unexpected host invariant failure")

    with pytest.raises(RuntimeError, match="unexpected host invariant failure"):
        pipeline.generate_team_policy(
            {"task_orchestrator_plan": _task_plan()},
            artifact_id="attempt_07",
            coordination_mode="async",
            validator=broken_host_validator,
        )

    assert len(client.calls) == 2
    assert [record["role"] for record in pipeline.records] == ["coder", "coder"]


def test_async_repair_validation_does_not_hide_host_runtime_error(tmp_path):
    client = FakeSharedQwenVL(
        [
            _async_robot_draft("front"),
            _async_robot_draft("back"),
            _async_team_review_payload(),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    validation_count = 0

    def validator_with_late_host_bug(candidate):
        nonlocal validation_count
        assert "robot_scripts" in candidate
        validation_count += 1
        if validation_count > 1:
            raise RuntimeError("late host invariant failure")

    with pytest.raises(RuntimeError, match="late host invariant failure"):
        pipeline.generate_team_policy(
            {"task_orchestrator_plan": _task_plan()},
            artifact_id="attempt_07",
            coordination_mode="async",
            validator=validator_with_late_host_bug,
        )

    assert validation_count == 2
    assert len(client.calls) == 3
    assert [record["role"] for record in pipeline.records] == [
        "coder",
        "coder",
        "code_repair",
    ]
    assert pipeline.records[-1]["validation_error"] == (
        "RuntimeError: late host invariant failure"
    )


def test_async_coder_phase_ids_follow_committed_orchestrator_phase(tmp_path):
    front = _async_robot_draft("front")
    back = _async_robot_draft("back")
    front["phase_id"] = "coder_front_corner_alias"
    back["phase_id"] = "coder_back_corner_alias"
    client = FakeSharedQwenVL(
        [front, back, _async_team_review_payload()]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    plan = _task_plan()

    policy = pipeline.generate_team_policy(
        {"task_orchestrator_plan": plan},
        artifact_id="attempt_08",
        coordination_mode="async",
    )

    expected_phase = plan["current_phase_id"]
    assert {
        script["phase_id"]
        for script in policy["robot_scripts"].values()
    } == {expected_phase}
    assert {
        assignment["phase_id"]
        for assignment in policy["assignments"].values()
    } == {expected_phase}
    assert (
        "front.phase_id: coder_front_corner_alias -> initial_straight "
        "(Task Orchestrator authority)"
        in pipeline.records[0]["host_schema_normalizations"]
    )
    assert (
        "back.phase_id: coder_back_corner_alias -> initial_straight "
        "(Task Orchestrator authority)"
        in pipeline.records[1]["host_schema_normalizations"]
    )
    review_prompt = client.calls[2]["user"]
    assert review_prompt.count(
        f'"phase_id": "{expected_phase}"'
    ) >= 2


def test_async_coder_sync_groups_follow_committed_orchestrator_subtask(tmp_path):
    front = _async_robot_draft("front")
    back = _async_robot_draft("back")
    front["synchronization_group"] = "front_invented_group"
    back["synchronization_group"] = "back_invented_group"
    client = FakeSharedQwenVL(
        [front, back, _async_team_review_payload()]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    plan = _task_plan()

    policy = pipeline.generate_team_policy(
        {"task_orchestrator_plan": plan},
        artifact_id="attempt_08",
        coordination_mode="async",
    )

    expected_group = plan["current_subtask"]["subtask_id"]
    assert {
        script["synchronization_group"]
        for script in policy["robot_scripts"].values()
    } == {expected_group}
    assert {
        assignment["synchronization_group"]
        for assignment in policy["assignments"].values()
    } == {expected_group}
    assert any(
        "front.synchronization_group: front_invented_group "
        f"-> {expected_group}"
        in item
        for item in pipeline.records[0]["host_schema_normalizations"]
    )
    assert any(
        "back.synchronization_group: back_invented_group "
        f"-> {expected_group}"
        in item
        for item in pipeline.records[1]["host_schema_normalizations"]
    )


def test_task_orchestrator_pivot_role_retry_preserves_selected_direction(tmp_path):
    response = _task_plan()
    client = FakeSharedQwenVL([response, response])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )
    validation_count = 0

    def reject_first_role_assignment(_payload):
        nonlocal validation_count
        validation_count += 1
        if validation_count == 1:
            raise ModelRoleError(
                "coordinated_pivot role geometry requires front=inner and "
                "back=outer for turn_direction=clockwise; front/back are "
                "immutable robot IDs"
            )

    pipeline.call_json_role(
        "task_orchestrator",
        {},
        artifact_id="attempt_09",
        validator=reject_first_role_assignment,
    )

    retry_prompt = client.calls[1]["user"]
    assert "This is a role-metadata repair, not a new strategy choice" in retry_prompt
    assert "Preserve the coordinated_pivot turn direction" in retry_prompt
    assert "front=inner and back=outer" in retry_prompt


def test_async_coordination_draft_cannot_replace_reference_with_hold(tmp_path):
    invalid_front = _async_robot_draft("front")
    invalid_front["code"] = (
        "def policy(api):\n"
        "    api.hold_position(hold_s=2.0)\n"
        "    api.stop_all()\n"
    )
    client = FakeSharedQwenVL(
        [
            invalid_front,
            _async_robot_draft("front"),
            _async_robot_draft("back"),
            _async_team_review_payload(),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    policy = pipeline.generate_team_policy(
        {
            "task_orchestrator_plan": _task_plan(),
            "committed_execution_strategy": {
                "primitive": "coordinated_alignment",
                "coordination_mode": "async",
            },
        },
        artifact_id="attempt_07",
        coordination_mode="async",
    )

    assert policy["team_review"]["decision"] == "approve_drafts"
    assert pipeline.records[0]["valid"] is False
    assert "must call that exact primitive" in pipeline.records[0][
        "validation_error"
    ]


def test_async_payload_contact_intent_is_repaired_by_coder_before_team_review(
    tmp_path,
):
    invalid_back = _pivot_robot_draft("back")
    invalid_back["contact_intent"] = "none"
    client = FakeSharedQwenVL(
        [
            _pivot_robot_draft("front"),
            invalid_back,
            _pivot_robot_draft("back"),
            _async_team_review_payload(),
        ]
    )
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    policy = pipeline.generate_team_policy(
        {
            "task_orchestrator_plan": _pivot_task_plan(),
            "committed_execution_strategy": {
                "primitive": "coordinated_pivot",
                "coordination_mode": "async",
            },
        },
        artifact_id="attempt_06",
        coordination_mode="async",
    )

    assert policy["robot_scripts"]["back"]["contact_intent"] == (
        "maintain_payload_support"
    )
    assert [record["role"] for record in pipeline.records] == [
        "coder",
        "coder",
        "coder",
        "code_repair",
    ]
    assert pipeline.records[1]["valid"] is False
    assert "payload-cooperative skill" in pipeline.records[1]["validation_error"]
    assert "contact_intent to exactly" in client.calls[2]["user"]
    assert "Do not add contact_intent as an api method argument" in client.calls[2][
        "user"
    ]
    assert '"contact_intent": "none"' not in client.calls[3]["user"]


def test_async_payload_contact_intent_stall_never_reaches_code_repair(tmp_path):
    invalid_back = _pivot_robot_draft("back")
    invalid_back["contact_intent"] = "none"
    client = FakeSharedQwenVL(
        [
            _pivot_robot_draft("front"),
            invalid_back,
            invalid_back,
        ]
    )
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    with pytest.raises(
        ModelRoleStallError,
        match="coder failed structured validation",
    ):
        pipeline.generate_team_policy(
            {
                "task_orchestrator_plan": _pivot_task_plan(),
                "committed_execution_strategy": {
                    "primitive": "coordinated_pivot",
                    "coordination_mode": "async",
                },
            },
            artifact_id="attempt_06",
            coordination_mode="async",
        )

    assert [record["role"] for record in pipeline.records] == [
        "coder",
        "coder",
        "coder",
    ]
    assert all(record["role"] != "code_repair" for record in pipeline.records)


def test_formal_initial_policy_is_plan_guided_without_candidate_selector(tmp_path):
    client = FakeSharedQwenVL([_task_plan(), _sync_linear_team_policy_payload()])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    orchestrator = ModelTaskOrchestrator(
        client=client,
        coordination_layer_enabled=True,
        role_pipeline=pipeline,
    )
    observation = ObservationBundle(
        runtime_state={
            "payload": {
                "present": True,
                "active": True,
                "contacts": {
                    "support_robots": ["front", "back"],
                    "floor": False,
                    "walls": [],
                },
            }
        }
    )

    policy = orchestrator.generate_initial_policy(
        _task_for_plan(
            _task_plan(),
            task_id="plan_guided",
            route_discovery=True,
        ),
        observation,
        attempt=1,
    )

    assert policy.script_id == "sync_linear_team"
    assert [record["role"] for record in pipeline.records] == [
        "task_orchestrator",
        "coder",
    ]
    assert pipeline.current_task_plan["current_phase_id"] == "initial_straight"


def test_formal_initial_target_turn_reaches_coder_and_executable_contract(tmp_path):
    plan = _pivot_task_plan()
    plan["global_plan"] = [
        {
            "phase_id": "approach_destination",
            "abstract_goal": "approach the declared final target",
            "completion_evidence": "payload reaches the placement region",
            "status": "active",
        },
        {
            "phase_id": "deposit_payload",
            "abstract_goal": "place and release the payload",
            "completion_evidence": "payload is physically target-supported",
            "status": "pending",
        },
    ]
    plan["current_phase_id"] = "approach_destination"
    plan["current_subtask"]["navigation"] = {
        "reference": "final_target",
        "arrival_condition": "face and approach the final target",
    }
    plan["current_subtask"]["strategy"]["semantic_objective"] = "clockwise"
    plan["current_subtask"]["team_assignment"] = {
        "front": {
            "role": "inner",
            "objective": "remain as the inner pivot support",
        },
        "back": {
            "role": "outer",
            "objective": "perform the forward outer arc",
        },
    }
    plan["rationale"] = (
        "the live target is behind the shared robot heading, so reduce the "
        "declared target-heading error before translating"
    )

    def pivot_draft(robot_id: str) -> dict:
        role = "inner" if robot_id == "front" else "outer"
        code = (
            "def policy(api):\n"
            "    api.coordinated_pivot("
            "turn_direction='clockwise', angle_deg=30.0, "
            "travel_direction='forward', duration_s=8.0, "
            f"role='{role}', swing_radius_m=0.8)\n"
            "    api.stop_all()\n"
        )
        return {
            "robot_id": robot_id,
            "script_id": f"{robot_id}_target_turn",
            "role": role,
            "objective": (
                "remain as the inner pivot support"
                if role == "inner"
                else "perform the forward outer arc"
            ),
            "phase_id": "approach_destination",
            "contact_intent": "maintain_payload_support",
            "synchronization_group": "target_turn_team",
            "coordination_mode": "async",
            "code": code,
        }

    runtime_state = {
        "checkpoint": {"checkpoint_id": "accepted_step_16"},
        "payload": {
            "present": True,
            "active": True,
            "x": 0.6303562549231021,
            "y": 7.769384449186844,
            "contacts": {
                "support_robots": ["front", "back"],
                "floor": False,
                "walls": [],
            },
        },
        "target": {
            "available": True,
            "center_x_m": -1.5,
            "center_y_m": 7.6,
        },
        "robots": {
            "front": {
                "x": 0.390760207866787,
                "y": 7.250483826365288,
                "yaw_deg": -11.7706,
            },
            "back": {
                "x": 0.5900008451466768,
                "y": 8.349486054931441,
                "yaw_deg": -10.9781,
            },
        },
    }
    target_turn_projection = _fixed_global_camera_heading_projection(runtime_state)
    _set_turn_comparison(
        plan,
        direction="clockwise",
        selected_heading=target_turn_projection["clockwise"]["screen_heading"],
        rejected_heading=target_turn_projection["counterclockwise"]["screen_heading"],
        decision_basis="clockwise reduces the live final-target heading error",
    )
    client = FakeSharedQwenVL(
        [
            plan,
            pivot_draft("front"),
            pivot_draft("back"),
            _async_team_review_payload(),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    orchestrator = ModelTaskOrchestrator(
        client=client,
        coordination_layer_enabled=True,
        role_pipeline=pipeline,
        formal_eval=True,
    )

    policy = orchestrator.generate_initial_policy(
        _task_for_plan(
            plan,
            task_id="generic_target_turn",
            target="place payload on the declared target",
            route_discovery=True,
            route_progress_gates=(),
        ),
        ObservationBundle(runtime_state=runtime_state),
        attempt=1,
    )

    local = policy.verification_contract["local_navigation"]
    assert local["reference"] == "final_target"
    assert local["required_turn_direction"] == "clockwise"
    assert local["heading_alignment_required"] is True
    assert "local_navigation_contract" in client.calls[1]["user"]
    assert [record["role"] for record in pipeline.records] == [
        "task_orchestrator",
        "coder",
        "coder",
        "code_repair",
    ]


def test_formal_evolution_uses_orchestrator_subtask_not_candidate_menu(tmp_path):
    client = FakeSharedQwenVL([_task_plan(), _sync_linear_team_policy_payload()])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = _task_plan()
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        task=_task_for_plan(_task_plan(), route_discovery=True),
        route_discovery=True,
        coordination_layer_enabled=True,
        role_pipeline=pipeline,
    )
    observation = ObservationBundle(
        runtime_state={
            "checkpoint": {"checkpoint_id": "accepted_step_03"},
            "payload": {
                "present": True,
                "active": True,
                "contacts": {
                    "support_robots": ["front", "back"],
                    "floor": False,
                    "walls": [],
                },
            },
        }
    )
    report = EvaluationReport(
        success=True,
        failure_type="none",
        metrics={"task_complete": 0.0},
    )
    current = _policy()
    decision = agent.decide_next_step(
        current,
        report,
        observation,
        ExecutionTrace(script_id=current.script_id, attempt=1, status="success"),
        checkpoint_id="accepted_step_03",
        physically_dispatched=True,
    )

    evolved, _spec = agent.evolve(
        current,
        report,
        observation,
        controller_decision=decision,
    )

    assert evolved.script_id == "sync_linear_team"
    assert [record["role"] for record in pipeline.records] == [
        "task_orchestrator",
        "coder",
    ]
    orchestrator_call = json.loads(
        (
            tmp_path
            / "model_roles"
            / "task_orchestrator_attempt_02.json"
        ).read_text(encoding="utf-8")
    )
    planning_prompt = orchestrator_call["user_prompt"]
    assert '"previous_subtask_outcome": "accepted_progress"' in planning_prompt
    assert '"last_accepted_policy":' in planning_prompt
    assert '"last_accepted_rollout":' in planning_prompt
    assert '"last_rejected_policy":' not in planning_prompt
    assert '"last_rejected_rollout":' not in planning_prompt
    assert '"last_rejected_failure_analysis":' not in planning_prompt
    assert "ACCEPT_SETUP keeps the same active subtask" in planning_prompt
    assert '"accepted_subtask_history": {' in planning_prompt
    assert '"current_subtask": {' not in planning_prompt.partition(
        '"previous_task_plan":'
    )[2].partition('"retrieved_strategy_memory":')[0]
    decision_artifact = json.loads(
        (tmp_path / "model_evolution_decision_01.json").read_text(
            encoding="utf-8"
        )
    )
    assert decision_artifact["decision_path"] == "plan_guided_model_roles"


def test_role_retry_keeps_all_prior_field_corrections_active(tmp_path):
    client = FakeSharedQwenVL([_task_plan(), _task_plan(), _task_plan()])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=2,
    )
    validation_count = 0

    def alternating_validator(_candidate):
        nonlocal validation_count
        validation_count += 1
        if validation_count == 1:
            raise ModelRoleError(
                "semantic_objective for api.coordinated_pivot must be one of "
                "['clockwise', 'counterclockwise']; got 'align team heading'"
            )
        if validation_count == 2:
            raise ModelRoleError(
                "selection rationale contradicts current_runtime_state: "
                "the current payload has no wall contact"
            )

    plan = pipeline.plan_task(
        {},
        artifact_id="attempt_13",
        validator=alternating_validator,
    )

    assert plan["current_subtask"]["strategy"]["semantic_objective"] == "forward"
    third_prompt = client.calls[2]["user"]
    assert "semantic_objective for api.coordinated_pivot" in third_prompt
    assert "contradicts current_runtime_state" in third_prompt
    assert "Do not paraphrase that token" in third_prompt
    assert "Treat current_state_authority as the present checkpoint" in third_prompt


def test_wall_contact_avoidance_is_not_a_false_current_collision_claim():
    _validate_selection_current_state_claims(
        selected_method="coordinated_linear_transport",
        reason=(
            "Use synchronized backward transport while maintaining bilateral "
            "support and avoiding wall contact."
        ),
        runtime_state={
            "payload": {
                "supporting_robots": ["front", "back"],
                "wall_contacts": [],
            }
        },
    )


def test_orchestrator_retry_requires_robot_ids_as_assignment_keys(tmp_path):
    invalid = _task_plan()
    invalid["current_subtask"]["team_assignment"] = {
        "inner": {"role": "inner", "objective": "hold pivot"},
        "outer": {"role": "outer", "objective": "follow arc"},
    }
    valid = _task_plan()
    client = FakeSharedQwenVL([invalid, valid])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    plan = pipeline.plan_task({}, artifact_id="attempt_15")

    assert set(plan["current_subtask"]["team_assignment"]) == {"front", "back"}
    retry_prompt = client.calls[1]["user"]
    assert "object keys with exactly front and back" in retry_prompt
    assert "Never use role names such as inner and outer as object keys" in (
        retry_prompt
    )


def test_formal_orchestrator_keeps_completed_turn_advisory(tmp_path):
    pivot = _task_plan()
    pivot["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_pivot",
            "semantic_objective": "clockwise",
            "coordination_mode": "async",
        }
    )
    pivot["current_subtask"]["team_assignment"] = {
        "front": {"role": "inner", "objective": "pivot clockwise"},
        "back": {"role": "outer", "objective": "pivot clockwise"},
    }
    _set_turn_comparison(
        pivot,
        direction="clockwise",
        selected_heading="left",
        rejected_heading="right",
    )
    valid = _task_plan()
    runtime_state = {
        "turn_episode": {
            "active": False,
            "status": "passage_aligned",
            "purpose": "route_turn",
            "direction": "clockwise",
            "roles": {"front": "inner", "back": "outer"},
            "translation_pending": True,
            "latest_passage_posture": "lateral_aligned",
        }
    }
    client = FakeSharedQwenVL([pivot])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    plan = pipeline.plan_task(
        {},
        artifact_id="attempt_24",
        validator=lambda candidate: _validate_orchestrator_plan_strategy(
            candidate,
            allowed_sdk_methods={
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
            runtime_state=runtime_state,
            model_owns_route=True,
        ),
    )

    assert plan["current_subtask"]["strategy"]["primitive"] == "coordinated_pivot"
    assert len(client.calls) == 1

    strict_client = FakeSharedQwenVL([pivot, valid])
    strict_pipeline = ModelRolePipeline(client=strict_client, output_dir=tmp_path)
    strict_plan = strict_pipeline.plan_task(
        {},
        artifact_id="attempt_25",
        validator=lambda candidate: _validate_orchestrator_plan_strategy(
            candidate,
            allowed_sdk_methods={
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
            runtime_state=runtime_state,
            model_owns_route=False,
        ),
    )

    assert (
        strict_plan["current_subtask"]["strategy"]["primitive"]
        == "coordinated_linear_transport"
    )
    assert len(strict_client.calls) == 2
    retry_prompt = strict_client.calls[1]["user"]
    assert "End the pivot subtask" in retry_prompt
    assert "Do not copy the accepted pivot" in retry_prompt


def test_formal_orchestrator_rejects_signed_turn_comparison_conflict():
    plan = _task_plan()
    plan["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "clockwise",
        "coordination_mode": "async",
    }
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "inner", "objective": "hold inner pivot"},
        "back": {"role": "outer", "objective": "follow outer arc"},
    }
    plan["rationale"] = (
        "Both clockwise and counterclockwise have immediate wall clearance, "
        "so choose clockwise."
    )
    runtime_state = {
        "robots": {
            "front": {"x": 0.0, "y": -0.5, "yaw_deg": 0.0},
            "back": {"x": 0.0, "y": 0.5, "yaw_deg": 0.0},
        }
    }

    with pytest.raises(ModelRoleError, match="STRUCTURED_TURN_COMPARISON_CONFLICT"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods={"coordinated_pivot"},
            runtime_state=runtime_state,
            model_owns_route=True,
        )

    with pytest.raises(ModelRoleError, match="STRUCTURED_TURN_COMPARISON_CONFLICT"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods={"coordinated_pivot"},
            runtime_state=runtime_state,
            model_owns_route=False,
        )


def test_formal_orchestrator_accepts_connected_passage_turn_projection():
    plan = _task_plan()
    plan["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "counterclockwise",
        "coordination_mode": "async",
    }
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "outer", "objective": "follow outer arc"},
        "back": {"role": "inner", "objective": "hold inner pivot"},
    }
    plan["rationale"] = (
        "The host-projected candidates are clockwise -> up-left and "
        "counterclockwise -> down-left. The immediately connected visible "
        "passage continues down-left: selected counterclockwise -> down-left "
        "follows the passage; rejected clockwise -> up-left because it leaves "
        "that corridor."
    )
    runtime_state = {
        "robots": {
            "front": {"x": 0.0, "y": -0.5, "yaw_deg": 0.0},
            "back": {"x": 0.0, "y": 0.5, "yaw_deg": 0.0},
        }
    }
    projection = _fixed_global_camera_heading_projection(runtime_state)
    _set_turn_comparison(
        plan,
        direction="counterclockwise",
        selected_heading=projection["counterclockwise"]["screen_heading"],
        rejected_heading=projection["clockwise"]["screen_heading"],
    )

    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods={"coordinated_pivot"},
        runtime_state=runtime_state,
        model_owns_route=True,
    )


def test_formal_orchestrator_accepts_explicit_turn_comparison_without_magic_words():
    """The semantic contract must not depend on hidden lexical passwords."""

    plan = _task_plan()
    plan["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "counterclockwise",
        "coordination_mode": "async",
    }
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "outer", "objective": "follow outer arc"},
        "back": {"role": "inner", "objective": "hold inner pivot"},
    }
    # This is the geometry-grounded rationale returned by GPT in the V23
    # diagnostic run. It satisfies the documented selected/rejected form even
    # though it does not contain the implementation words "project" or
    # "candidate".
    plan["rationale"] = (
        "selected counterclockwise -> down-left follows the immediately "
        "connected visible passage upward in the fixed global view; rejected "
        "clockwise -> up-left because it tracks the same corner but not the "
        "connected upward passage opening as directly."
    )
    runtime_state = {
        "robots": {
            "front": {"x": 0.0, "y": -0.5, "yaw_deg": 0.0},
            "back": {"x": 0.0, "y": 0.5, "yaw_deg": 0.0},
        }
    }
    projection = _fixed_global_camera_heading_projection(runtime_state)
    _set_turn_comparison(
        plan,
        direction="counterclockwise",
        selected_heading=(
            "screen-heading "
            + projection["counterclockwise"]["screen_heading"].replace("-", "_")
        ),
        rejected_heading=projection["clockwise"]["screen_heading"].replace(
            "-", "_"
        ),
    )
    plan["rationale"] = "Rationale wording is deliberately unrelated to parsing."

    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods={"coordinated_pivot"},
        runtime_state=runtime_state,
        model_owns_route=True,
    )


def test_formal_formation_only_reference_rejects_route_motion_but_allows_alignment():
    route_plan = _task_plan()
    route_plan["current_subtask"]["navigation"]["reference"] = "formation_only"

    with pytest.raises(ModelRoleError, match="pure team-repair declaration"):
        _validate_orchestrator_plan_strategy(
            route_plan,
            allowed_sdk_methods={
                "coordinated_alignment",
                "coordinated_linear_transport",
            },
            model_owns_route=True,
        )

    alignment_plan = json.loads(json.dumps(route_plan))
    alignment_plan["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_alignment",
            "semantic_objective": "longitudinal",
            "coordination_mode": "async",
        }
    )
    alignment_plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "reference", "objective": "hold position"},
        "back": {"role": "corrector", "objective": "reduce stagger"},
    }

    _validate_orchestrator_plan_strategy(
        alignment_plan,
        allowed_sdk_methods={
            "coordinated_alignment",
            "coordinated_linear_transport",
        },
        model_owns_route=True,
    )


def test_formal_final_target_reference_requires_target_approach_phase():
    premature = _task_plan()
    premature["current_subtask"]["navigation"]["reference"] = "final_target"

    with pytest.raises(ModelRoleError, match="final_target.*premature"):
        _validate_orchestrator_plan_strategy(
            premature,
            allowed_sdk_methods={"coordinated_linear_transport"},
            model_owns_route=True,
        )

    target_approach = json.loads(json.dumps(premature))
    target_approach["global_plan"] = [
        {
            "phase_id": "target_approach",
            "abstract_goal": "approach the final target",
            "completion_evidence": "payload enters the placement region",
            "status": "active",
        },
        {
            "phase_id": "payload_placement",
            "abstract_goal": "place and release the payload",
            "completion_evidence": "payload is supported by the target",
            "status": "pending",
        },
    ]
    target_approach["current_phase_id"] = "target_approach"

    _validate_orchestrator_plan_strategy(
        target_approach,
        allowed_sdk_methods={"coordinated_linear_transport"},
        final_target_navigation_authorized=True,
        model_owns_route=True,
    )


def test_formal_pivot_keeps_model_roles_while_nonformal_requires_host_geometry():
    plan = _task_plan()
    plan["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "clockwise",
        "coordination_mode": "async",
    }
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "outer", "objective": "follow outer arc"},
        "back": {"role": "inner", "objective": "hold inner pivot"},
    }
    runtime_state = {
        "robots": {
            "front": {"x": 0.0, "y": -0.5, "yaw_deg": 0.0},
            "back": {"x": 0.0, "y": 0.5, "yaw_deg": 0.0},
        }
    }
    projection = _fixed_global_camera_heading_projection(runtime_state)
    _set_turn_comparison(
        plan,
        direction="clockwise",
        selected_heading=projection["clockwise"]["screen_heading"],
        rejected_heading=projection["counterclockwise"]["screen_heading"],
    )

    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods={"coordinated_pivot"},
        runtime_state=runtime_state,
        model_owns_route=True,
    )

    with pytest.raises(ModelRoleError, match="role geometry requires"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods={"coordinated_pivot"},
            runtime_state=runtime_state,
            model_owns_route=False,
        )


def test_formal_active_turn_rejects_linear_escape_but_completed_turn_allows_it():
    plan = _task_plan()
    active_turn = {
        "turn_episode": {
            "active": True,
            "status": "turning",
            "purpose": "route_turn",
            "direction": "clockwise",
            "roles": {"front": "inner", "back": "outer"},
            "switch_allowed": False,
        }
    }

    with pytest.raises(ModelRoleError, match="route-turn episode is unfinished"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods={"coordinated_linear_transport"},
            runtime_state=active_turn,
            model_owns_route=True,
        )

    completed_turn = json.loads(json.dumps(active_turn))
    completed_turn["turn_episode"].update(
        {"active": False, "status": "passage_aligned"}
    )
    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods={"coordinated_linear_transport"},
        runtime_state=completed_turn,
        model_owns_route=True,
    )


def test_formal_orchestrator_rejects_projection_choice_conflict():
    plan = _task_plan()
    plan["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "clockwise",
        "coordination_mode": "async",
    }
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "inner", "objective": "hold inner pivot"},
        "back": {"role": "outer", "objective": "follow outer arc"},
    }
    plan["rationale"] = (
        "The host-projected candidates are clockwise -> up-left and "
        "counterclockwise -> down-left. The connected passage is down-left: "
        "selected counterclockwise -> down-left follows the passage; rejected "
        "clockwise -> up-left because it leaves the corridor."
    )
    runtime_state = {
        "robots": {
            "front": {"x": 0.0, "y": -0.5, "yaw_deg": 0.0},
            "back": {"x": 0.0, "y": 0.5, "yaw_deg": 0.0},
        }
    }
    projection = _fixed_global_camera_heading_projection(runtime_state)
    _set_turn_comparison(
        plan,
        direction="counterclockwise",
        selected_heading=projection["counterclockwise"]["screen_heading"],
        rejected_heading=projection["clockwise"]["screen_heading"],
    )

    with pytest.raises(ModelRoleError, match="STRUCTURED_TURN_COMPARISON_CONFLICT"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods={"coordinated_pivot"},
            runtime_state=runtime_state,
            model_owns_route=True,
        )

    with pytest.raises(ModelRoleError, match="STRUCTURED_TURN_COMPARISON_CONFLICT"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods={"coordinated_pivot"},
            runtime_state=runtime_state,
            model_owns_route=False,
        )


def test_formal_final_target_turn_accepts_target_projection_evidence():
    plan = _task_plan()
    plan["global_plan"] = [
        {
            "phase_id": "target_approach",
            "abstract_goal": "approach the final target",
            "completion_evidence": "payload reaches the target placement region",
            "status": "active",
        },
        {
            "phase_id": "payload_placement",
            "abstract_goal": "place and release the payload",
            "completion_evidence": "payload is supported by the target",
            "status": "pending",
        },
    ]
    plan["current_phase_id"] = "target_approach"
    plan["current_subtask"]["navigation"] = {
        "reference": "final_target",
        "arrival_condition": "reach the target placement region",
    }
    plan["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "counterclockwise",
        "coordination_mode": "async",
    }
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "outer", "objective": "follow outer arc"},
        "back": {"role": "inner", "objective": "hold inner pivot"},
    }
    runtime_state = {
        "payload": {"active": True, "x": -0.86, "y": 7.78},
        "target": {
            "available": True,
            "center_x_m": -1.5,
            "center_y_m": 7.6,
        },
        "robots": {
            "front": {"x": -0.29, "y": 7.66, "yaw_deg": 92.0},
            "back": {"x": -1.03, "y": 7.63, "yaw_deg": 92.0},
        },
        "turn_episode": {
            "active": False,
            "status": "passage_aligned",
            "purpose": "route_turn",
            "direction": "clockwise",
            "translation_pending": True,
        },
    }
    projection = _fixed_global_camera_heading_projection(runtime_state)
    _set_turn_comparison(
        plan,
        direction="counterclockwise",
        selected_heading=projection["counterclockwise"]["screen_heading"],
        rejected_heading=projection["clockwise"]["screen_heading"],
        decision_basis="counterclockwise reduces the live final-target heading error",
    )
    plan["rationale"] = (
        "The host-projected candidates are clockwise -> "
        f"{projection['clockwise']['screen_heading']} and counterclockwise -> "
        f"{projection['counterclockwise']['screen_heading']}. The live final "
        "target lies along the selected counterclockwise -> "
        f"{projection['counterclockwise']['screen_heading']} candidate; rejected "
        f"clockwise -> {projection['clockwise']['screen_heading']} because it "
        "turns away from the target."
    )

    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods={"coordinated_pivot"},
        runtime_state=runtime_state,
        final_target_navigation_authorized=True,
        model_owns_route=True,
    )


def test_completed_turn_sync_repair_feedback_requires_alignment(tmp_path):
    invalid = _task_plan()
    invalid["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "counterclockwise",
        "formation_strategy": "target reorientation",
        "coordination_mode": "async",
        "synchronization_intent": "turn together",
    }
    invalid["current_subtask"]["team_assignment"] = {
        "front": {"role": "outer", "objective": "swing"},
        "back": {"role": "inner", "objective": "pivot"},
    }
    _set_turn_comparison(
        invalid,
        direction="counterclockwise",
        selected_heading="down-left",
        rejected_heading="up-left",
    )
    valid = json.loads(json.dumps(invalid))
    valid["current_subtask"]["strategy"] = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "formation_strategy": "relative formation repair",
        "coordination_mode": "async",
        "synchronization_intent": "let the lagging robot catch up",
    }
    valid["current_subtask"]["team_assignment"] = {
        "front": {"role": "reference", "objective": "hold"},
        "back": {"role": "corrector", "objective": "catch up"},
    }
    valid["current_subtask"]["turn_comparison"] = json.loads(
        json.dumps(_task_plan()["current_subtask"]["turn_comparison"])
    )
    client = FakeSharedQwenVL([invalid, valid])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def validator(candidate: dict) -> None:
        primitive = candidate["current_subtask"]["strategy"]["primitive"]
        if primitive != "coordinated_alignment":
            raise ModelRoleError(
                "the formation synchronization probe failed repeatedly and "
                "the payload already completed this route turn; the repair "
                "requires coordinated_alignment at this checkpoint, not linear "
                "transport or another pivot"
            )

    plan = pipeline.plan_task(
        {},
        artifact_id="attempt_43",
        validator=validator,
    )

    assert plan["current_subtask"]["strategy"]["primitive"] == (
        "coordinated_alignment"
    )
    retry_prompt = client.calls[1]["user"]
    assert "exactly one executable repair family" in retry_prompt
    assert "Do not emit coordinated_linear_transport or coordinated_pivot" in (
        retry_prompt
    )
    assert "Set the current primitive to coordinated_linear_transport" not in (
        retry_prompt
    )


def test_completed_turn_without_alternative_linear_feedback_requires_alignment(
    tmp_path,
):
    linear = _task_plan()
    pivot = json.loads(json.dumps(linear))
    pivot["current_subtask"]["strategy"] = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "clockwise",
        "formation_strategy": "loaded pivot",
        "coordination_mode": "async",
        "synchronization_intent": "one shared turn checkpoint",
    }
    pivot["current_subtask"]["team_assignment"] = {
        "front": {"role": "inner", "objective": "hold"},
        "back": {"role": "outer", "objective": "swing"},
    }
    _set_turn_comparison(
        pivot,
        direction="clockwise",
        selected_heading="up-left",
        rejected_heading="down-left",
    )
    alignment = json.loads(json.dumps(linear))
    alignment["current_subtask"]["strategy"] = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "lateral",
        "formation_strategy": "relative formation repair",
        "coordination_mode": "async",
        "synchronization_intent": "correct relative offset",
    }
    alignment["current_subtask"]["team_assignment"] = {
        "front": {"role": "reference", "objective": "hold"},
        "back": {"role": "corrector", "objective": "align"},
    }
    client = FakeSharedQwenVL([linear, pivot, alignment])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    def validator(candidate: dict) -> None:
        primitive = candidate["current_subtask"]["strategy"]["primitive"]
        if primitive == "coordinated_linear_transport":
            raise ModelRoleError(
                "strategy_mutation has no alternative "
                "coordinated_linear_transport objective satisfying the combined "
                "trusted constraints"
            )
        if primitive == "coordinated_pivot":
            raise ModelRoleError(
                "the proposed strategy violates the current live turn episode: "
                "the payload already completed this route turn; select linear "
                "transport or a bounded alignment before another turn"
            )

    plan = pipeline.plan_task(
        {},
        artifact_id="attempt_44",
        validator=validator,
    )

    assert plan["current_subtask"]["strategy"]["primitive"] == (
        "coordinated_alignment"
    )
    retry_prompt = client.calls[2]["user"]
    assert "exactly one executable repair family" in retry_prompt
    assert "Do not emit coordinated_linear_transport or coordinated_pivot" in (
        retry_prompt
    )
    assert "Set the current primitive to coordinated_linear_transport" not in (
        retry_prompt
    )


def test_evolution_manager_normalizes_parameter_bearing_calls(tmp_path):
    response = {
        **_manager("strategy_mutation"),
        "mutate": [
            "api.carry_forward(distance_m=2.0, speed_mps=0.5, ramp_s=1.5)"
        ],
        "inheritance": [
            "api.coordinated_pivot(turn_direction='counterclockwise', "
            "angle_deg=30.0, duration_s=8.0)"
        ],
    }
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    decision = pipeline.decide_mutation({}, artifact_id="attempt_12")

    assert decision["mutate"] == ["maneuver family", "semantic objective"]
    assert decision["inheritance"] == [
        "global mission",
        "current route phase",
        "accepted checkpoint progress",
        "payload support",
        "physical safety",
    ]
    assert len(client.calls) == 1
    artifact = json.loads(
        (
            tmp_path / "model_roles" / "evolution_manager_attempt_12.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["raw_parsed_response"] == response
    assert artifact["host_schema_normalizations"] == [
        "mutate[0]: executable call -> symbolic strategy",
        "inheritance[0]: executable call -> symbolic strategy",
        "mutate: concrete strategy targets -> abstract mutation dimensions",
        "inheritance: concrete parent contract -> safe mission invariants",
    ]


def test_evolution_manager_normalizes_self_contradictory_strategy_contract(
    tmp_path,
):
    response = {
        **_manager("strategy_mutation"),
        "mutate": [
            "primitive: coordinated_alignment",
            "semantic objective: longitudinal",
            "coordination mode: async",
        ],
        "forbidden": [
            "change primitive to coordinated_lateral_transport",
            "change coordination mode to sync",
            "change temporary roles to sync",
        ],
        "inheritance": [
            "global mission",
            "current route phase",
            "accepted checkpoint progress",
            "payload support",
            "physical safety",
        ],
    }
    pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([response]),
        output_dir=tmp_path,
    )

    decision = pipeline.decide_mutation({}, artifact_id="attempt_47")

    assert decision["mutate"] == [
        "maneuver family",
        "semantic objective",
        "formation strategy",
    ]
    assert decision["forbidden"] == [
        "repeat the exact checkpoint-local physical counterexample",
        "violate payload support or physical safety",
    ]
    assert decision["inheritance"] == response["inheritance"]


def test_evolution_manager_names_parameter_but_leaves_value_to_later_role(
    tmp_path,
):
    invalid = {
        **_manager("parameter_tuning"),
        "mutate": ["angle_deg=30.0 reduced to 15.0"],
    }
    valid = {
        **_manager("parameter_tuning"),
        "mutate": ["angle_deg"],
    }
    client = FakeSharedQwenVL([invalid, valid])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    decision = pipeline.decide_mutation({}, artifact_id="attempt_20")

    assert decision["mutation_level"] == "parameter_tuning"
    assert decision["mutate"] == ["angle_deg"]
    assert len(client.calls) == 2
    retry_prompt = client.calls[1]["user"]
    assert "remove all old values, proposed values" in retry_prompt


def test_mutation_operator_does_not_treat_inheritance_prose_as_executable(tmp_path):
    valid = _pivot_strategy_mutation()
    valid.update(
        {
            "mutation_level": "parameter_tuning",
            "reason": "tune angle_deg while preserving the selected pivot strategy",
            "changed_dimensions": ["angle_deg"],
            "inheritance": "preserve the signed pivot strategy and roles",
        }
    )
    invalid = {
        **valid,
        "inheritance": [
            "preserve duration_s=8.0",
            "preserve swing_radius_m=0.8",
        ],
    }
    client = FakeSharedQwenVL([invalid])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    mutation = pipeline.mutate_strategy({}, artifact_id="attempt_21")

    assert mutation["changed_dimensions"] == ["angle_deg"]
    assert mutation["inheritance"] == invalid["inheritance"]
    assert len(client.calls) == 1
    artifact = json.loads(
        (tmp_path / "model_roles" / "mutation_operator_attempt_21.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["host_schema_normalizations"] == []


def test_mutation_operator_may_cite_rejected_parameter_value_in_reason(tmp_path):
    response = _pivot_strategy_mutation()
    response.update(
        {
            "mutation_level": "parameter_tuning",
            "reason": (
                "The rejected angle_deg=30 caused instability; tune angle_deg "
                "while preserving duration_s, swing_radius_m, and the signed "
                "pivot strategy."
            ),
            "changed_dimensions": ["angle_deg"],
            "inheritance": [
                "accepted checkpoint progress: accepted_step_27",
                "preserve the signed pivot strategy and roles",
            ],
        }
    )
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    mutation = pipeline.mutate_strategy({}, artifact_id="attempt_21")

    assert mutation["changed_dimensions"] == ["angle_deg"]
    assert "accepted_step_27" in mutation["inheritance"][0]
    assert len(client.calls) == 1


def test_mutation_operator_preserves_parameter_words_in_role_objective_prose(
    tmp_path,
):
    response = _pivot_strategy_mutation()
    response["parent_strategy"]["team_assignment"] = {
        "front": {
            "role": "reference",
            "objective": "hold with zero delta",
        },
        "back": {
            "role": "corrector",
            "objective": "move with delta=0.1189",
        },
    }
    response["child_strategy"]["team_assignment"] = {
        "front": {
            "role": "inner",
            "objective": "hold with zero angle_deg",
        },
        "back": {
            "role": "outer",
            "objective": "turn with angle_deg=30",
        },
    }
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    mutation = pipeline.mutate_strategy({}, artifact_id="attempt_07")

    assert mutation["parent_strategy"]["team_assignment"]["front"][
        "objective"
    ] == "hold with zero delta"
    assert mutation["child_strategy"]["team_assignment"]["back"][
        "objective"
    ] == "turn with angle_deg=30"
    assert pipeline.records[0]["host_schema_normalizations"] == []


def test_strategy_mutation_reason_may_name_failed_parameter_semantics(tmp_path):
    response = _pivot_strategy_mutation()
    response["reason"] = (
        "The rejected formation correction used a longitudinal delta, so switch "
        "to the selected symbolic pivot strategy without choosing action values."
    )
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    mutation = pipeline.mutate_strategy({}, artifact_id="attempt_08")

    assert mutation["mutation_level"] == "strategy_mutation"
    assert len(client.calls) == 1


def test_strategy_mutation_accepts_parameter_name_inside_symbolic_enum(tmp_path):
    response = _pivot_strategy_mutation()
    response["child_strategy"].update(
        {
            "primitive": "coordinated_alignment",
            "semantic_objective": "lateral",
            "formation_strategy": "corrector_delta_based",
            "synchronization_intent": "repair_relative_formation_error",
            "team_assignment": {
                "front": {
                    "role": "reference",
                    "objective": "maintain lateral reference",
                },
                "back": {
                    "role": "corrector",
                    "objective": "apply relative formation correction",
                },
            },
        }
    )
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    mutation = pipeline.mutate_strategy({}, artifact_id="attempt_53")

    assert mutation["child_strategy"]["formation_strategy"] == (
        "corrector_delta_based"
    )
    assert len(client.calls) == 1


def test_strategy_mutation_preserves_parameter_words_in_sync_intent_prose(tmp_path):
    response = _pivot_strategy_mutation()
    response["child_strategy"].update(
        {
            "primitive": "coordinated_alignment",
            "semantic_objective": "lateral",
            "formation_strategy": "async coordinated_alignment",
            "coordination_mode": "async",
            "synchronization_intent": (
                "corrector applies nonzero delta while reference applies zero"
            ),
            "team_assignment": {
                "front": {
                    "role": "reference",
                    "objective": "serve as the temporary formation reference",
                },
                "back": {
                    "role": "corrector",
                    "objective": "apply the selected relative formation correction",
                },
            },
        }
    )
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    mutation = pipeline.mutate_strategy({}, artifact_id="attempt_54")

    assert mutation["child_strategy"]["synchronization_intent"] == (
        "corrector applies nonzero delta while reference applies zero"
    )
    assert len(client.calls) == 1
    assert pipeline.records[0]["host_schema_normalizations"] == []


def test_async_coder_receives_explicit_turn_repair_envelope(tmp_path):
    front = _pivot_robot_draft("front")
    back = _pivot_robot_draft("back")
    for payload in (front, back):
        payload["code"] = payload["code"].replace(
            "angle_deg=30.0", "angle_deg=15.0"
        )
    client = FakeSharedQwenVL([front, back, _async_team_review_payload()])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    pipeline.generate_team_policy(
        {
            "task_orchestrator_plan": _task_plan(),
            "turn_execution_repair_contract": {
                "previous_angle_deg": 30.0,
                "maximum_next_angle_deg": 15.0,
                "minimum_angle_deg": 2.0,
                "previous_duration_s": 8.0,
            },
        },
        artifact_id="attempt_21",
        coordination_mode="async",
    )

    assert len(client.calls) == 3
    assert all(
        "angle_deg no greater than 15.0" in call["user"]
        and "keep duration_s=8.0" in call["user"]
        for call in client.calls
    )


def test_code_repair_retry_requires_full_pivot_envelope(tmp_path):
    response = _async_team_policy_payload()
    client = FakeSharedQwenVL([response, response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    validation_count = 0

    def pivot_validator(_candidate):
        nonlocal validation_count
        validation_count += 1
        if validation_count == 1:
            raise ModelRoleError(
                "coordination primitive requires matching front/back "
                "travel_direction; got {'front': None, 'back': 'forward'}"
            )

    repaired = pipeline.repair_code(
        {},
        artifact_id="attempt_13_team_review",
        validator=pivot_validator,
    )

    assert repaired["script_id"] == "async_team"
    retry_prompt = client.calls[1]["user"]
    assert 'travel_direction=\\"forward\\"' in retry_prompt
    assert "Do not omit travel_direction or swing_radius_m" in retry_prompt


def test_orchestrator_pivot_requires_exact_inner_outer_roles():
    plan = _task_plan()
    plan["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_pivot",
            "semantic_objective": "counterclockwise",
            "coordination_mode": "async",
        }
    )
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "pivot_inner", "objective": "hold the pivot"},
        "back": {"role": "swing_outer", "objective": "follow the arc"},
    }

    with pytest.raises(ModelRoleError, match="exactly one inner and one outer"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods=frozenset({"coordinated_pivot"}),
        )

    plan["current_subtask"]["team_assignment"]["front"]["role"] = "inner"
    plan["current_subtask"]["team_assignment"]["back"]["role"] = "outer"
    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods=frozenset({"coordinated_pivot"}),
    )


def test_orchestrator_alignment_requires_reference_corrector_roles():
    plan = _task_plan()
    plan["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_alignment",
            "semantic_objective": "yaw",
            "coordination_mode": "async",
        }
    )
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "payload_stabilizer", "objective": "hold heading"},
        "back": {"role": "payload_stabilizer", "objective": "correct heading"},
    }

    with pytest.raises(ModelRoleError, match="one reference and one corrector"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods=frozenset({"coordinated_alignment"}),
        )

    plan["current_subtask"]["team_assignment"]["front"]["role"] = "reference"
    plan["current_subtask"]["team_assignment"]["back"]["role"] = "corrector"
    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods=frozenset({"coordinated_alignment"}),
    )


def test_orchestrator_cannot_immediately_reverse_accepted_linear_progress():
    plan = _task_plan()
    plan["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_linear_transport",
            "semantic_objective": "right",
            "coordination_mode": "sync",
        }
    )
    continuation = {
        "active": True,
        "primitive": "coordinated_linear_transport",
        "semantic_objective": "left",
        "blocked_immediate_inverse": "right",
    }

    with pytest.raises(
        ModelRoleError,
        match="exact prohibited inverse.*Current wall-clear linear alternatives",
    ):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods=frozenset({"coordinated_linear_transport"}),
            accepted_linear_continuation_contract=continuation,
            host_wall_clearance={
                "open_translation_directions": ["backward", "left", "right"]
            },
        )

    plan["current_subtask"]["strategy"]["semantic_objective"] = "left"
    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods=frozenset({"coordinated_linear_transport"}),
        accepted_linear_continuation_contract=continuation,
    )


def test_strategy_mutation_cannot_select_target_away_final_transport():
    parent = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "formation_strategy": "bilateral_support",
        "coordination_mode": "async",
        "synchronization_intent": "repair",
        "team_assignment": {
            "front": {"role": "reference", "objective": "hold"},
            "back": {"role": "corrector", "objective": "align"},
        },
    }
    child = {
        "primitive": "coordinated_linear_transport",
        "semantic_objective": "forward",
        "formation_strategy": "bilateral_support",
        "coordination_mode": "sync",
        "synchronization_intent": "target_approach",
        "team_assignment": {
            "front": {"role": "mover", "objective": "move"},
            "back": {"role": "mover", "objective": "move"},
        },
    }
    payload = {
        "mutation_level": "strategy_mutation",
        "child_strategy": child,
    }

    with pytest.raises(ModelRoleError, match="points away from the physical target"):
        _validate_plan_guided_mutation(
            payload,
            manager_decision={"mutation_level": "strategy_mutation"},
            parent_strategy=parent,
            allowed_sdk_methods=frozenset({
                "coordinated_alignment",
                "coordinated_linear_transport",
            }),
            trusted_body_frame_target_projection={
                "target_capture_active": True,
                "candidate_target_alignment": {
                    "forward": -0.962,
                    "backward": 0.962,
                    "left": 0.273,
                    "right": -0.273,
                },
            },
        )


def test_orchestrator_retry_copies_checkpoint_formation_gate(tmp_path):
    invalid = _task_plan()
    corrected = json.loads(json.dumps(invalid))
    corrected["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_alignment",
            "semantic_objective": "longitudinal",
            "coordination_mode": "async",
        }
    )
    corrected["current_subtask"]["team_assignment"] = {
        "front": {"role": "reference", "objective": "hold"},
        "back": {"role": "corrector", "objective": "catch up"},
    }
    gate = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "coordination_mode": "async",
        "team_roles": {"front": "reference", "back": "corrector"},
    }
    client = FakeSharedQwenVL([invalid, corrected])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=1,
    )

    def validate(candidate):
        strategy = candidate["current_subtask"]["strategy"]
        roles = {
            robot_id: assignment["role"]
            for robot_id, assignment in candidate["current_subtask"][
                "team_assignment"
            ].items()
        }
        if (
            strategy["primitive"] != gate["primitive"]
            or strategy["semantic_objective"] != gate["semantic_objective"]
            or strategy["coordination_mode"] != gate["coordination_mode"]
            or roles != gate["team_roles"]
        ):
            raise ModelRoleError(
                "the current trusted checkpoint requires one route-neutral formation "
                "repair before further route motion"
            )

    result = pipeline.plan_task(
        {"host_formation_gate": gate},
        artifact_id="attempt_06",
        validator=validate,
    )

    assert result["current_subtask"]["strategy"]["primitive"] == (
        "coordinated_alignment"
    )
    retry_prompt = client.calls[1]["user"]
    assert "mandatory local physical precondition" in retry_prompt
    assert "Copy host_formation_gate.primitive" in retry_prompt
    assert "does not choose the later route direction" in retry_prompt


def test_orchestrator_rejects_linear_during_unfinished_route_turn():
    plan = _task_plan()
    runtime_state = {
        "turn_episode": {
            "active": True,
            "status": "switch_allowed",
            "purpose": "route_turn",
            "direction": "counterclockwise",
            "switch_allowed": True,
            "translation_pending": False,
        }
    }

    with pytest.raises(ModelRoleError, match="route-turn episode is unfinished"):
        _validate_orchestrator_plan_strategy(
            plan,
            allowed_sdk_methods=frozenset(
                {"coordinated_linear_transport", "coordinated_pivot"}
            ),
            runtime_state=runtime_state,
        )

    plan["current_subtask"]["strategy"].update(
        {
            "primitive": "coordinated_pivot",
            "semantic_objective": "clockwise",
            "coordination_mode": "async",
        }
    )
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "inner", "objective": "hold the pivot"},
        "back": {"role": "outer", "objective": "swing forward"},
    }
    _validate_orchestrator_plan_strategy(
        plan,
        allowed_sdk_methods=frozenset(
            {"coordinated_linear_transport", "coordinated_pivot"}
        ),
        runtime_state=runtime_state,
    )


def test_strategy_mutation_rejects_linear_escape_from_unfinished_turn():
    parent = _pivot_strategy_mutation()["child_strategy"]
    linear_child = _task_plan()["current_subtask"]["strategy"]
    payload = {
        "mutation_level": "strategy_mutation",
        "child_strategy": {
            **linear_child,
            "team_assignment": _task_plan()["current_subtask"]["team_assignment"],
        },
    }
    runtime_state = {
        "turn_episode": {
            "active": True,
            "status": "switch_allowed",
            "purpose": "route_turn",
            "direction": "counterclockwise",
            "switch_allowed": True,
            "translation_pending": False,
        }
    }

    with pytest.raises(
        ModelRoleError,
        match="semantic_objective='clockwise'",
    ):
        _validate_plan_guided_mutation(
            payload,
            manager_decision={"mutation_level": "strategy_mutation"},
            parent_strategy=parent,
            allowed_sdk_methods={
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
            runtime_state=runtime_state,
        )

    opposite = _pivot_strategy_mutation()
    opposite["child_strategy"]["semantic_objective"] = "clockwise"
    _validate_plan_guided_mutation(
        opposite,
        manager_decision={"mutation_level": "strategy_mutation"},
        parent_strategy=parent,
        allowed_sdk_methods={
            "coordinated_linear_transport",
            "coordinated_pivot",
        },
        runtime_state=runtime_state,
    )


def test_strategy_replan_rejects_pivot_wrapped_in_stale_linear_subtask():
    approved = _pivot_strategy_mutation()["child_strategy"]
    stale = _task_plan()
    stale["current_subtask"]["strategy"] = {
        key: value for key, value in approved.items() if key != "team_assignment"
    }
    stale["current_subtask"]["team_assignment"] = approved["team_assignment"]

    with pytest.raises(
        ModelRoleError,
        match="describe the approved maneuver",
    ):
        _validate_strategy_mutation_orchestrator_replan(
            stale,
            previous_task_plan=_task_plan(),
            approved_strategy=approved,
        )

    _validate_strategy_mutation_orchestrator_replan(
        _pivot_task_plan(),
        previous_task_plan=_task_plan(),
        approved_strategy=approved,
    )

    # A model-only retry can load the already-correct partial replan as its
    # previous plan. Idempotent approved semantics must remain valid.
    _validate_strategy_mutation_orchestrator_replan(
        _pivot_task_plan(),
        previous_task_plan=_pivot_task_plan(),
        approved_strategy=approved,
    )


def test_strategy_replan_accepts_retreat_as_backward_completion_semantics():
    plan = _task_plan()
    strategy = plan["current_subtask"]["strategy"]
    strategy.update(
        {
            "primitive": "coordinated_linear_transport",
            "semantic_objective": "backward",
            "coordination_mode": "sync",
        }
    )
    strategy["team_assignment"] = {
        "front": {"role": "mover", "objective": "translate backward"},
        "back": {"role": "mover", "objective": "translate backward"},
    }
    plan["current_subtask"].update(
        {
            "objective": "execute backward transport",
            "guidance": "use synchronized linear transport",
            "completion_condition": "payload retreats clear of the obstruction",
            "team_assignment": strategy["team_assignment"],
        }
    )
    plan["rationale"] = "reverse travel is open in the trusted state"

    _validate_strategy_mutation_orchestrator_replan(
        plan,
        previous_task_plan=plan,
        approved_strategy=strategy,
    )


def test_strategy_replan_accepts_role_local_alignment_objectives():
    approved = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "coordination_mode": "async",
        "formation_strategy": "async coordinated alignment",
        "synchronization_intent": "corrector moves while reference stays fixed",
        "team_assignment": {
            "front": {"role": "reference"},
            "back": {"role": "corrector"},
        },
    }
    plan = _task_plan()
    plan["current_subtask"].update(
        {
            "objective": "perform longitudinal formation alignment",
            "guidance": "correct the payload support offset before transport",
            "strategy": approved,
            "team_assignment": {
                "front": {
                    "role": "reference",
                    "objective": "remain stationary while the partner adjusts",
                },
                "back": {
                    "role": "corrector",
                    "objective": "move forward to reduce support center offset",
                },
            },
            "completion_condition": "formation alignment is physically accepted",
        }
    )
    plan["rationale"] = "longitudinal correction addresses the measured desync"

    _validate_strategy_mutation_orchestrator_replan(
        plan,
        previous_task_plan=_task_plan(),
        approved_strategy=approved,
    )


def test_strategy_replan_inherits_only_locked_route_identity(tmp_path):
    parent = _task_plan()
    parent_navigation = parent["current_subtask"]["navigation"]
    parent_navigation.update(
        {
            "validated_route_segment_id": "cross_passage",
            "route_reference_locked": True,
            "route_observation_captured_at_unix": 1234.5,
        }
    )
    approved = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "yaw",
        "coordination_mode": "async",
        "formation_strategy": "coordinated alignment yaw correction",
        "synchronization_intent": "corrector moves while reference stays fixed",
        "team_assignment": {
            "front": {"role": "corrector"},
            "back": {"role": "reference"},
        },
    }
    candidate = _task_plan()
    candidate["current_subtask"].update(
        {
            "objective": "correct loaded team yaw alignment",
            "guidance": "apply one bounded yaw correction",
            "navigation": {
                "reference": "final_target",
                "arrival_condition": "formation is repaired for final approach",
                "world_direction_xy": [0.0, 0.0],
                "completion_world_direction_xy": [0.0, 0.0],
            },
            "strategy": {
                key: value
                for key, value in approved.items()
                if key != "team_assignment"
            },
            "team_assignment": {
                "front": {
                    "role": "corrector",
                    "objective": "apply the yaw correction",
                },
                "back": {
                    "role": "reference",
                    "objective": "hold as reference during yaw correction",
                },
            },
            "completion_condition": "yaw alignment is physically accepted",
        }
    )
    candidate["rationale"] = "yaw correction repairs the measured route drift"
    pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([candidate]),
        output_dir=tmp_path,
        response_retries=0,
    )

    def normalize(plan: dict) -> None:
        _inherit_strategy_replan_navigation_contract(plan, parent_plan=parent)

    def validate(plan: dict) -> None:
        _validate_strategy_mutation_orchestrator_replan(
            plan,
            previous_task_plan=parent,
            approved_strategy=approved,
        )

    result = pipeline.plan_task(
        {"previous_task_plan": parent},
        artifact_id="attempt_20",
        normalizer=normalize,
        validator=validate,
    )

    navigation = result["current_subtask"]["navigation"]
    assert navigation["reference"] == "final_target"
    assert navigation["world_direction_xy"] == [0.0, 0.0]
    assert navigation["completion_world_direction_xy"] == [0.0, 0.0]
    assert navigation["validated_route_segment_id"] == "cross_passage"
    assert navigation["route_reference_locked"] is True
    assert navigation["route_observation_captured_at_unix"] == 1234.5
    assert result["current_subtask"]["strategy"] == {
        key: value for key, value in approved.items() if key != "team_assignment"
    }
    assert pipeline.records[0]["valid"] is True
    assert pipeline.records[0]["host_schema_normalizations"] == [
        (
            "strategy replan navigation: inherited checkpoint route identity fields "
            "validated_route_segment_id, route_reference_locked, "
            "route_observation_captured_at_unix; child reference, direction vectors, "
            "local geometry repair, primitive, mode, and roles were preserved"
        )
    ]


@pytest.mark.parametrize("reference", ["final_target", "formation_only"])
def test_strategy_replan_preserves_synthetic_navigation_semantics(reference):
    # Synthetic input exercises the same inheritance and idempotence contract;
    # recorded provider outputs and their original audit stay local.
    plan = _task_plan()
    expected = {
        "reference": reference,
        "world_direction_xy": [0.0, 1.0],
        "completion_world_direction_xy": [-1.0, 0.0],
        "arrival_condition": "verified synthetic milestone",
    }
    plan["current_subtask"]["navigation"] = dict(expected)
    parent = _task_plan()
    parent["current_subtask"]["navigation"].update({
        "validated_route_segment_id": "trusted_segment",
        "route_reference_locked": True,
        "route_observation_captured_at_unix": 1234.5,
    })
    _inherit_strategy_replan_navigation_contract(plan, parent_plan=parent)
    once = json.loads(json.dumps(plan))
    _inherit_strategy_replan_navigation_contract(plan, parent_plan=parent)
    navigation = plan["current_subtask"]["navigation"]
    assert all(navigation[key] == value for key, value in expected.items())
    assert navigation["validated_route_segment_id"] == "trusted_segment"
    assert plan == once


@pytest.mark.parametrize("reference", ["final_target", "formation_only"])
def test_zero_vector_semantic_navigation_is_valid(reference, tmp_path):
    plan = _task_plan()
    plan["current_subtask"]["navigation"] = {
        "reference": reference,
        "arrival_condition": "the semantic subtask is physically verified",
        "world_direction_xy": [0.0, 0.0],
        "completion_world_direction_xy": [0.0, 0.0],
    }
    pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([plan]),
        output_dir=tmp_path,
        response_retries=0,
    )

    result = pipeline.plan_task({}, artifact_id="attempt_01")

    assert result["current_subtask"]["navigation"]["reference"] == reference
    assert pipeline.records[0]["normalization_audit"]["raw_response_preserved"] is True


def test_formal_evolution_rejects_stale_payload_reinitialization_plan(tmp_path):
    client = FakeSharedQwenVL(
        [
            _stale_payload_initialization_plan(),
            _task_plan(),
            _sync_linear_team_policy_payload(),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = _stale_payload_initialization_plan()
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        task=_task_for_plan(_task_plan(), route_discovery=True),
        route_discovery=True,
        coordination_layer_enabled=True,
        role_pipeline=pipeline,
    )
    observation = ObservationBundle(
        runtime_state={
            "checkpoint": {"checkpoint_id": "accepted_step_01"},
            "payload": {
                "present": True,
                "active": True,
                "contacts": {
                    "support_robots": ["front", "back"],
                    "floor": False,
                    "walls": [],
                },
            },
        }
    )
    report = EvaluationReport(
        success=True,
        failure_type="none",
        metrics={"task_complete": 0.0},
    )
    current = _policy()
    decision = agent.decide_next_step(
        current,
        report,
        observation,
        ExecutionTrace(script_id=current.script_id, attempt=1, status="success"),
        checkpoint_id="accepted_step_01",
        physically_dispatched=True,
    )

    evolved, _spec = agent.evolve(
        current,
        report,
        observation,
        controller_decision=decision,
    )

    assert evolved.script_id == "sync_linear_team"
    assert [record["role"] for record in pipeline.records] == [
        "task_orchestrator",
        "task_orchestrator",
        "coder",
    ]
    first_plan_call = json.loads(
        (
            tmp_path
            / "model_roles"
            / "task_orchestrator_attempt_02.json"
        ).read_text(encoding="utf-8")
    )
    assert first_plan_call["valid"] is False
    assert "physical lifecycle forbids api.place_payload_on_arms" in (
        first_plan_call["validation_error"]
    )
    assert '"previous_task_plan": {}' in first_plan_call["user_prompt"]
    retry_plan_call = json.loads(
        (
            tmp_path
            / "model_roles"
            / "task_orchestrator_attempt_02_call_02.json"
        ).read_text(encoding="utf-8")
    )
    assert "remove stale deferred/inactive payload claims" in (
        retry_plan_call["user_prompt"]
    )


def test_failed_subtask_uses_manager_mutation_then_three_call_async_coding(
    tmp_path,
):
    strategy_replan = _pivot_task_plan()
    client = FakeSharedQwenVL(
        [
            _manager("strategy_mutation"),
            _pivot_strategy_mutation(),
            strategy_replan,
            _pivot_robot_draft("front"),
            _pivot_robot_draft("back"),
            _async_team_review_payload(),
        ]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = _task_plan()
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        task=_task_for_plan(_task_plan(), route_discovery=True),
        route_discovery=True,
        coordination_layer_enabled=True,
        role_pipeline=pipeline,
    )
    observation = ObservationBundle(
        runtime_state={
            "checkpoint": {"checkpoint_id": "accepted_step_03"},
            "payload": {
                "present": True,
                "active": True,
                "contacts": {
                    "support_robots": ["front", "back"],
                    "floor": False,
                    "walls": [],
                },
            },
        }
    )
    report = EvaluationReport(
        success=False,
        failure_type="route_regression",
        metrics={"payload_step_displacement_xy_m": 0.4},
    )
    current = _policy()
    decision = agent.decide_next_step(
        current,
        report,
        observation,
        ExecutionTrace(script_id=current.script_id, attempt=1, status="success"),
        checkpoint_id="accepted_step_03",
        physically_dispatched=True,
    )

    evolved, _spec = agent.evolve(
        current,
        report,
        observation,
        controller_decision=decision,
    )

    assert evolved.script_id == "counterclockwise_corner_pivot"
    assert [record["role"] for record in pipeline.records] == [
        "evolution_manager",
        "mutation_operator",
        "task_orchestrator",
        "coder",
        "coder",
        "code_repair",
    ]
    manager_call = json.loads(
        (
            tmp_path
            / "model_roles"
            / "evolution_manager_attempt_01.json"
        ).read_text(encoding="utf-8")
    )
    manager_prompt = manager_call["user_prompt"]
    assert '"previous_subtask_outcome": "rejected"' in manager_prompt
    assert '"current_trusted_observation":' in manager_prompt
    assert '"current_state_authority":' in manager_prompt
    assert '"rejected_endpoint_policy":' in manager_prompt
    assert '"rejected_endpoint_rollout":' in manager_prompt
    assert '"rejected_endpoint_failure_analysis":' in manager_prompt
    mutation_call = json.loads(
        (
            tmp_path
            / "model_roles"
            / "mutation_operator_attempt_02.json"
        ).read_text(encoding="utf-8")
    )
    mutation_prompt = mutation_call["user_prompt"]
    assert '"current_trusted_observation":' in mutation_prompt
    assert '"current_state_authority":' in mutation_prompt
    assert '"rejected_endpoint_policy":' in mutation_prompt
    assert '"rejected_endpoint_rollout":' in mutation_prompt
    assignment = json.loads(
        (
            tmp_path
            / "model_roles"
            / "subtask_assignment_attempt_02.json"
        ).read_text(encoding="utf-8")
    )
    assert assignment["strategy_source"] == "host_locked_mutation_operator_child"
    assert assignment["current_subtask"]["objective"] == (
        "execute one bounded counterclockwise pivot"
    )
    assert assignment["current_subtask"]["navigation"]["world_direction_xy"] == [
        0.0,
        1.0,
    ]
    replan_call = json.loads(
        (
            tmp_path
            / "model_roles"
            / "task_orchestrator_attempt_02.json"
        ).read_text(encoding="utf-8")
    )
    assert '"locked_child_strategy":' in replan_call["user_prompt"]
    locked_child = json.loads(
        (
            tmp_path
            / "model_roles"
            / "locked_mutation_child_attempt_02.json"
        ).read_text(encoding="utf-8")
    )
    assert locked_child["immutable_executable_fields_downstream"] is True
    assert locked_child["child_strategy"]["semantic_objective"] == (
        "counterclockwise"
    )


def test_memory_role_demotes_parameter_bearing_strategy_genome(tmp_path):
    leaked = {
        **_memory_update(),
        "child_strategy": {
            "primitive": "coordinated_linear_transport",
            "speed_mps": 0.5,
        },
    }
    client = FakeSharedQwenVL([leaked])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path, response_retries=0)

    proposal = pipeline.propose_memory_update({}, artifact_id="attempt_03")

    assert proposal["promote"] is False
    artifact = json.loads(
        (tmp_path / "model_roles" / "memory_update_attempt_03.json").read_text()
    )
    assert artifact["host_schema_normalizations"] == [
        "promote: true -> false (host memory-transferability gate)"
    ]


def test_nonpromoted_memory_proposal_may_explain_why_it_is_scene_specific(tmp_path):
    rejected = {
        **_memory_update(promote=False),
        "learned_rule": "This checkpoint-specific route claim is not reusable.",
    }
    pipeline = ModelRolePipeline(
        client=FakeSharedQwenVL([rejected]),
        output_dir=tmp_path,
        response_retries=0,
    )

    proposal = pipeline.propose_memory_update({}, artifact_id="attempt_04")

    assert proposal["promote"] is False


def test_strategy_memory_requires_physics_before_promoting_model_genome(tmp_path):
    store = StrategyMemoryStore(tmp_path / "strategy.json")
    proposal = _memory_update()

    assert not store.update_from_rollout(
        {
            "dry_run": True,
            "coordination_mode": "sync",
            "executed_skill": "coordinated_linear_transport",
            "verifier": {"step_accepted": True, "failure_type": "none"},
        },
        model_proposal=proposal,
    )
    assert store.update_from_rollout(
        {
            "kind": "physical_rollout_finalized",
            "dry_run": False,
            "physically_dispatched": True,
            "physical_transaction_completed": True,
            "script_id": "attempt_04",
            "attempt": 4,
            "coordination_mode": "sync",
            "executed_skill": "coordinated_linear_transport",
            "strategy_context": {"phase": "corridor_transport"},
            "verifier": {
                "step_accepted": True,
                "failure_type": "none",
                "checkpoint_before": "trusted_before",
                "checkpoint_after": "trusted_after",
            },
        },
        model_proposal=proposal,
    )
    assert store.specs[0]["strategy_genome"]["mutation_level"] == (
        "coordination_mutation"
    )


def test_strategy_memory_does_not_promote_accepted_setup(tmp_path):
    store = StrategyMemoryStore(tmp_path / "strategy.json")

    promoted = store.update_from_rollout(
        {
            "kind": "physical_rollout_finalized",
            "dry_run": False,
            "physically_dispatched": True,
            "physical_transaction_completed": True,
            "script_id": "setup_only",
            "attempt": 4,
            "coordination_mode": "async",
            "executed_skill": "coordinated_pivot",
            "strategy_context": {"phase": "corner_setup"},
            "verifier": {
                "step_accepted": True,
                "failure_type": "none",
                "transition_outcome": "ACCEPT_SETUP",
                "checkpoint_before": "trusted_before",
                "checkpoint_after": "trusted_after",
            },
        },
        model_proposal={
            **_memory_update(),
            "child_strategy": {
                "primitive": "coordinated_pivot",
                "coordination": "inner_outer",
            },
        },
    )

    assert not promoted
    assert store.specs == ()


def test_failed_rollout_does_not_promote_unexecuted_model_repair(tmp_path):
    store = StrategyMemoryStore(tmp_path / "strategy.json")
    proposal = {
        **_memory_update(),
        "child_strategy": {
            "primitive": "coordinated_pivot",
            "coordination": "inner_outer",
        },
    }

    assert store.update_from_rollout(
        {
            "kind": "physical_rollout_finalized",
            "dry_run": False,
            "physically_dispatched": True,
            "physical_transaction_completed": True,
            "script_id": "failed_linear",
            "attempt": 5,
            "coordination_mode": "sync",
            "executed_skill": "coordinated_linear_transport",
            "strategy_context": {"phase": "corridor_transport"},
            "verifier": {
                "step_accepted": False,
                "failure_type": "payload_drop",
                "checkpoint_before": "accepted_step_04",
                "checkpoint_after": "",
            },
        },
        model_proposal=proposal,
    )

    spec = store.specs[0]
    assert spec["kind"] == "verified_failure_counterexample"
    assert spec["observed_coordination_primitive"] == (
        "coordinated_linear_transport"
    )
    assert "strategy_genome" not in spec
    assert "coordinated_pivot" not in spec["decision"]


def test_accepted_memory_rejects_child_primitive_not_physically_executed(tmp_path):
    store = StrategyMemoryStore(tmp_path / "strategy.json")
    proposal = {
        **_memory_update(),
        "child_strategy": {
            "primitive": "coordinated_pivot",
            "coordination": "inner_outer",
        },
    }

    with pytest.raises(StrategyMemoryError, match="physically executed primitive"):
        store.update_from_rollout(
            {
                "kind": "physical_rollout_finalized",
                "dry_run": False,
                "physically_dispatched": True,
                "physical_transaction_completed": True,
                "script_id": "accepted_linear",
                "attempt": 6,
                "coordination_mode": "sync",
                "executed_skill": "coordinated_linear_transport",
                "strategy_context": {"phase": "corridor_transport"},
                "verifier": {
                    "step_accepted": True,
                    "failure_type": "none",
                    "checkpoint_before": "accepted_step_05",
                    "checkpoint_after": "accepted_step_06",
                },
            },
            model_proposal=proposal,
        )


def test_strategy_memory_rejects_model_genome_with_action_magnitude(tmp_path):
    store = StrategyMemoryStore(tmp_path / "strategy.json")
    proposal = {
        **_memory_update(),
        "learned_rule": "retry with distance_m reduced",
    }
    with pytest.raises(StrategyMemoryError, match="action parameters"):
        store.update_from_rollout(
            {
                "kind": "physical_rollout_finalized",
                "dry_run": False,
                "physically_dispatched": True,
                "physical_transaction_completed": True,
                "coordination_mode": "sync",
                "executed_skill": "coordinated_linear_transport",
                "strategy_context": {"phase": "corridor_transport"},
                "verifier": {"step_accepted": True, "failure_type": "none"},
            },
            model_proposal=proposal,
        )


def test_nontransferable_memory_role_is_demoted_and_does_not_block_rollout(tmp_path):
    leaked = {
        **_memory_update(),
        "child_strategy": {
            "primitive": "coordinated_linear_transport",
            "speed_mps": 0.5,
        },
    }
    client = FakeSharedQwenVL([leaked])
    pipeline = ModelRolePipeline(
        client=client,
        output_dir=tmp_path,
        response_retries=0,
    )
    store = StrategyMemoryStore(tmp_path / "strategy.json")

    payload = append_rollout_evidence(
        tmp_path / "raw.jsonl",
        "",
        dry_run=False,
        policy=_policy(),
        report=EvaluationReport(success=True, failure_type="none"),
        trace=ExecutionTrace(script_id="attempt_01", attempt=1, status="success"),
        strategy_store=store,
        strategy_phase="corridor_transport",
        role_pipeline=pipeline,
        physically_dispatched=True,
        physical_transaction_completed=True,
    )

    assert "model_memory_update_error" not in payload
    assert payload["memory_update"]["model_memory_update"]["promote"] is False
    assert store.specs == ()
    persisted = [
        json.loads(line)
        for line in (tmp_path / "raw.jsonl").read_text().splitlines()
    ]
    assert persisted[0]["kind"] == "physical_rollout_finalized"
    assert persisted[1]["kind"] == "memory_update_result"
    assert persisted[1]["model_memory_update"]["promote"] is False
    live_status = json.loads(
        (
            tmp_path / "model_roles" / "live_high_level_status.json"
        ).read_text(encoding="utf-8")
    )
    assert live_status["runtime_status"] == "memory_update_ready"
    assert live_status["active_model_role"] == "memory_update"
    assert live_status["details"]["promote"] is False


def test_merged_visual_role_returns_failure_analysis_in_one_model_call(tmp_path):
    frame = tmp_path / "global.png"
    frame.write_bytes(b"frame")
    client = FakeSharedQwenVL([_failure_analysis()])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    verifier = QwenVLBoxCarryVisualVerifier(
        client=client,
        role_pipeline=pipeline,
    )

    report = verifier.evaluate(
        task=BoxCarryTask(task_id="carry", target="target"),
        trace=ExecutionTrace(script_id="attempt_05", attempt=5, status="success"),
        visual=VisualObservation(source="global", frame_path=str(frame)),
        runtime_state={
            "payload": {
                "active": True,
                "z": 1.0,
                "contacts": {
                    "support_robots": ["front", "back"],
                    "floor": False,
                    "walls": [],
                },
            },
            "robots": {
                "front": {"yaw_rad": 0.0},
                "back": {"yaw_rad": 0.0},
            },
        },
        physics_report=EvaluationReport(
            success=False,
            failure_type="formation_desync",
        ),
    )

    assert len(client.calls) == 1
    assert report.evidence["failure_analysis"]["coordination_defect"] is True
    assert pipeline.records[0]["role"] == "vlm_failure_analyzer"


def test_visual_role_audits_harmless_qwen_schema_normalization(tmp_path):
    frame = tmp_path / "global.png"
    frame.write_bytes(b"frame")
    response = _failure_analysis()
    response["failure_analysis"] = {
        **response["failure_analysis"],
        "causal_chain": "no rollout action has occurred",
        "physical_evidence": "the payload is intentionally deferred",
        "visual_evidence": "both robots are upright",
        "reusable_context": "startup before payload activation",
        "strategy_defect": False,
        "coordination_defect": False,
        "magnitude_defect": False,
    }
    client = FakeSharedQwenVL([response])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    result = pipeline.analyze_rollout(
        {"physics_evidence": {"payload": {"active": False}}},
        artifact_id="initial_observation",
        image_path=frame,
    )

    assert len(client.calls) == 1
    assert result["failure_analysis"]["causal_chain"] == [
        "no rollout action has occurred"
    ]
    assert result["failure_analysis"]["strategy_defect"] is False
    assert result["failure_analysis"]["coordination_defect"] is False
    assert result["failure_analysis"]["magnitude_defect"] is False
    record = pipeline.records[0]
    assert record["raw_parsed_response"]["failure_analysis"]["causal_chain"] == (
        "no rollout action has occurred"
    )
    assert len(record["host_schema_normalizations"]) == 4


def test_agent_evolution_manager_overrides_host_repair_level_from_analysis(tmp_path):
    manager = _manager("parameter_tuning")
    manager.update(
        {
            "failure_hypothesis": (
                "the fixed-sync linear transaction exceeded its stable envelope"
            ),
            "mutate": ["distance_m"],
            "forbidden": ["change the semantic motion objective"],
        }
    )
    client = FakeSharedQwenVL([manager])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        role_pipeline=pipeline,
    )
    report = EvaluationReport(
        success=False,
        failure_type="payload_drop",
        evidence={
            "visual_verifier": {
                "evidence": {
                    "failure_analysis": {
                        **_failure_analysis()["failure_analysis"],
                        "primary_failure": "payload_drop",
                    }
                }
            }
        },
    )

    decision = agent.decide_next_step(
        _policy(distance_m=1.5),
        report,
        ObservationBundle(),
        ExecutionTrace(script_id="attempt_01", attempt=1, status="success"),
        checkpoint_id="trusted",
        physically_dispatched=True,
    )

    assert decision.action == "tune_parameter"
    assert decision.repair_level == "execution"
    assert (tmp_path / "model_evolution_manager_01.json").exists()


def test_minimum_linear_distance_failure_escalates_before_coder(tmp_path):
    parameter_manager = _manager("parameter_tuning")
    parameter_manager.update(
        {
            "failure_hypothesis": "the loaded segment was still too long",
            "mutate": ["distance_m"],
            "forbidden": ["change the semantic motion objective"],
        }
    )
    strategy_manager = _manager("strategy_mutation")
    strategy_manager.update(
        {
            "failure_hypothesis": (
                "the minimum loaded linear segment is unstable at this checkpoint"
            ),
            "preserve": [
                "the accepted checkpoint",
                "bilateral payload support",
            ],
            "mutate": [
                "semantic motion objective",
                "maneuver family",
            ],
            "forbidden": [
                "repeat the exact rejected transaction",
                "lose payload support",
            ],
            "inheritance": "retain the global mission and current route phase",
        }
    )
    client = FakeSharedQwenVL([parameter_manager, strategy_manager])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        role_pipeline=pipeline,
    )
    report = EvaluationReport(
        success=False,
        failure_type="payload_unstable",
        evidence={
            "visual_verifier": {
                "evidence": {
                    "failure_analysis": {
                        **_failure_analysis()["failure_analysis"],
                        "primary_failure": "payload_unstable",
                        "strategy_defect": False,
                        "coordination_defect": False,
                        "magnitude_defect": True,
                    }
                }
            }
        },
    )

    decision = agent.decide_next_step(
        _policy(distance_m=0.25),
        report,
        ObservationBundle(),
        ExecutionTrace(script_id="attempt_01", attempt=1, status="script_error"),
        checkpoint_id="accepted_step_13",
        physically_dispatched=True,
    )

    assert decision.action == "switch_strategy"
    assert decision.repair_level == "strategy"
    assert len(client.calls) == 2
    manager_artifact = json.loads(
        (tmp_path / "model_evolution_manager_01.json").read_text()
    )
    exhausted = manager_artifact["manager"]
    assert exhausted["mutation_level"] == "strategy_mutation"
    assert "parameter_envelope_exhausted" in client.calls[0]["user"]


def test_pair_spacing_abort_preserves_signed_pivot_and_tunes_angle(tmp_path):
    manager = _manager("parameter_tuning")
    manager.update(
        {
            "failure_hypothesis": "the loaded turn exceeded the spacing envelope",
            "evidence": ["robot_pair_too_far with bilateral support"],
            "preserve": [
                "coordinated_pivot clockwise",
                "front=inner and back=outer",
            ],
            "mutate": ["angle_deg"],
            "forbidden": ["change direction or roles"],
            "inheritance": "retain the verified signed pivot relation",
        }
    )
    client = FakeSharedQwenVL([manager])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        role_pipeline=pipeline,
    )

    def code(role: str) -> str:
        return (
            "def policy(api):\n"
            "    api.coordinated_pivot(turn_direction='clockwise', angle_deg=30.0, "
            f"duration_s=8.0, role='{role}', swing_radius_m=0.8)\n"
            "    api.stop_all()\n"
        )

    policy = PolicyScript(
        script_id="attempt_20",
        attempt=20,
        code=code("inner"),
        robot_scripts=(
            RobotPolicyScript(
                "front",
                "front_attempt_20",
                code("inner"),
                "turn_team",
                coordination_mode="async",
            ),
            RobotPolicyScript(
                "back",
                "back_attempt_20",
                code("outer"),
                "turn_team",
                coordination_mode="async",
            ),
        ),
    )
    report = EvaluationReport(
        success=False,
        failure_type="formation_desync",
        metrics={"robot_pair_xy_separation_m": 1.389},
        evidence={
            "payload": {
                "contacts": {
                    "floor": False,
                    "walls": [],
                    "support_forces": {
                        "normal_force_n_by_robot": {
                            "front": 1.7,
                            "back": 0.6,
                        }
                    },
                }
            },
            "visual_verifier": {
                "evidence": {
                    "failure_analysis": {
                        "outcome": "rejected",
                        "primary_failure": "formation_desync",
                        "causal_chain": ["payload tilt caused the failure"],
                        "physical_evidence": ["payload tilt 132 degrees"],
                        "visual_evidence": ["payload appears tilted"],
                        "strategy_defect": True,
                        "coordination_defect": True,
                        "magnitude_defect": True,
                        "reusable_context": [],
                        "confidence": 0.95,
                    }
                }
            },
        },
    )
    trace = ExecutionTrace(
        script_id="attempt_20",
        attempt=20,
        status="script_error",
        error="physical safety abort: robot_pair_too_far",
    )

    decision = agent.decide_next_step(
        policy,
        report,
        ObservationBundle(),
        trace,
        checkpoint_id="accepted_step_12",
        physically_dispatched=True,
    )

    assert decision.action == "tune_parameter"
    assert decision.repair_level == "execution"
    artifact = json.loads(
        (tmp_path / "model_evolution_manager_20.json").read_text()
    )
    analysis = artifact["failure_analysis"]
    assert analysis["source"] == (
        "physics_authoritative_robot_pair_spacing_override"
    )
    assert analysis["strategy_defect"] is False
    assert analysis["coordination_defect"] is False
    assert analysis["magnitude_defect"] is True
    assert analysis["authoritative_repair_contract"] == {
        "mutation_level": "parameter_tuning",
        "preserve_method": "coordinated_pivot",
        "preserve_turn_direction": "clockwise",
        "preserve_roles": {"front": "inner", "back": "outer"},
        "tune_only": "angle_deg",
    }


def test_host_locked_pivot_rejects_impossible_coordination_mutation(tmp_path):
    coordination_manager = _manager("coordination_mutation")
    strategy_manager = _manager("strategy_mutation")
    strategy_manager.update(
        {
            "mutate": ["maneuver primitive", "semantic objective"],
            "forbidden": ["swap host-owned pivot roles"],
        }
    )
    client = FakeSharedQwenVL([coordination_manager, strategy_manager])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        role_pipeline=pipeline,
    )

    def code(role: str) -> str:
        return (
            "def policy(api):\n"
            "    api.coordinated_pivot(turn_direction='counterclockwise', "
            "angle_deg=20.0, duration_s=8.0, "
            f"role='{role}', swing_radius_m=0.8)\n"
            "    api.stop_all()\n"
        )

    policy = PolicyScript(
        script_id="attempt_78",
        attempt=78,
        code=code("outer"),
        assignments=(
            RobotAssignment(
                robot_id="front",
                role="outer",
                objective="swing forward",
                synchronization_group="turn_team",
            ),
            RobotAssignment(
                robot_id="back",
                role="inner",
                objective="hold pivot",
                synchronization_group="turn_team",
            ),
        ),
        robot_scripts=(
            RobotPolicyScript(
                "front",
                "front_attempt_78",
                code("outer"),
                "turn_team",
                coordination_mode="async",
            ),
            RobotPolicyScript(
                "back",
                "back_attempt_78",
                code("inner"),
                "turn_team",
                coordination_mode="async",
            ),
        ),
    )
    report = EvaluationReport(
        success=False,
        failure_type="payload_drop",
        evidence={
            "visual_verifier": {
                "evidence": {
                    "failure_analysis": {
                        "outcome": "rejected",
                        "primary_failure": "one-sided support during pivot",
                        "causal_chain": ["front support was lost"],
                        "physical_evidence": ["payload floor contact"],
                        "visual_evidence": ["payload dropped"],
                        "strategy_defect": False,
                        "coordination_defect": True,
                        "magnitude_defect": False,
                        "reusable_context": [],
                        "confidence": 0.95,
                    }
                }
            }
        },
    )
    observation = ObservationBundle(
        runtime_state={
            "mean_heading_yaw_rad": 1.75,
            "robots": {
                "front": {"x": 0.8, "y": 7.5, "yaw_rad": 1.75},
                "back": {"x": -0.6, "y": 7.4, "yaw_rad": 1.75},
            },
            "payload": {"x": 0.1, "y": 7.45},
        }
    )

    decision = agent.decide_next_step(
        policy,
        report,
        observation,
        ExecutionTrace(
            script_id="attempt_78",
            attempt=78,
            status="script_error",
        ),
        checkpoint_id="accepted_step_53",
        physically_dispatched=True,
    )

    assert decision.action == "switch_strategy"
    assert decision.repair_level == "strategy"
    assert len(client.calls) == 2
    first_prompt = client.calls[0]["user"]
    assert "host_pivot_role_contract" in first_prompt
    assert "No coordination-only child exists" in first_prompt
    artifact = json.loads(
        (tmp_path / "model_evolution_manager_78.json").read_text()
    )
    assert artifact["manager"]["mutation_level"] == "strategy_mutation"


def test_parameter_tuning_task_plan_cannot_reselect_turn_direction():
    parent = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "clockwise",
        "formation_strategy": "async coordinated_pivot",
        "coordination_mode": "async",
        "synchronization_intent": "continuous temporary role coordination",
        "team_assignment": {
            "front": {"role": "inner", "objective": "pivot clockwise"},
            "back": {"role": "outer", "objective": "pivot clockwise"},
        },
    }
    plan = _task_plan()
    plan["current_subtask"]["strategy"] = {
        key: value
        for key, value in parent.items()
        if key != "team_assignment"
    }
    plan["current_subtask"]["team_assignment"] = parent["team_assignment"]

    _validate_parameter_tuning_task_plan(plan, parent_strategy=parent)

    plan["current_subtask"]["strategy"]["semantic_objective"] = (
        "counterclockwise"
    )
    with pytest.raises(
        ModelRoleError,
        match="parameter_tuning strategy contract requires",
    ):
        _validate_parameter_tuning_task_plan(plan, parent_strategy=parent)


def test_coordination_mutation_task_plan_cannot_reselect_alignment_axis():
    parent = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "formation_strategy": "async coordinated_alignment",
        "coordination_mode": "async",
        "synchronization_intent": "continuous temporary role coordination",
        "team_assignment": {
            "front": {"role": "reference", "objective": "hold"},
            "back": {"role": "corrector", "objective": "advance"},
        },
    }
    plan = _task_plan()
    plan["current_subtask"]["strategy"] = {
        key: value for key, value in parent.items() if key != "team_assignment"
    }
    plan["current_subtask"]["team_assignment"] = {
        "front": {"role": "corrector", "objective": "advance"},
        "back": {"role": "reference", "objective": "hold"},
    }
    plan["current_subtask"]["strategy"]["formation_strategy"] = (
        "front corrector with back reference"
    )

    _validate_coordination_mutation_task_plan(plan, parent_strategy=parent)
    assert plan["current_subtask"]["strategy"]["formation_strategy"] == (
        "async coordinated_alignment"
    )

    plan["current_subtask"]["strategy"]["semantic_objective"] = "yaw"
    with pytest.raises(
        ModelRoleError,
        match="coordination_mutation task-plan contract preserves",
    ):
        _validate_coordination_mutation_task_plan(plan, parent_strategy=parent)


def test_role_invocation_exhaustion_is_a_typed_five_response_stall(tmp_path):
    invalid = {"mutation_level": "not_a_level"}
    client = FakeSharedQwenVL([invalid] * 5)
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)

    with pytest.raises(ModelRoleStallError) as caught:
        pipeline.decide_mutation({}, artifact_id="attempt_09")

    assert caught.value.role == "evolution_manager"
    assert caught.value.response_count == 5
    assert len(pipeline.records) == 5
    assert all(record["valid"] is False for record in pipeline.records)


def test_prose_only_mutation_returns_machine_readable_effective_delta_feedback(
    tmp_path,
):
    parent = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "counterclockwise",
        "formation_strategy": "outer robot swings around inner robot",
        "coordination_mode": "async",
        "synchronization_intent": "continuous inner outer support",
        "team_assignment": {
            "front": {"role": "outer", "objective": "swing"},
            "back": {"role": "inner", "objective": "hold"},
        },
    }
    invalid_child = {
        **parent,
        "formation_strategy": "safer target-aware pivot near the pedestal",
        "synchronization_intent": "avoid another support loss",
    }
    valid_child = {
        **parent,
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "formation_strategy": "reference corrector alignment",
        "coordination_mode": "async",
        "synchronization_intent": "one reference and one corrector",
        "team_assignment": {
            "front": {"role": "corrector", "objective": "align"},
            "back": {"role": "reference", "objective": "hold"},
        },
    }

    def response(child: dict) -> dict:
        return {
            "mutation_level": "strategy_mutation",
            "reason": "repair the rejected near-target pivot",
            "parent_strategy": parent,
            "child_strategy": child,
            "changed_dimensions": ["formation strategy"],
            "inheritance": "accepted checkpoint and payload support",
        }

    client = FakeSharedQwenVL([response(invalid_child), response(valid_child)])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    contract = build_mutation_contract(parent)

    def validator(candidate: dict) -> None:
        validate_mutation_transition(
            candidate,
            parent_strategy=parent,
            contract=contract,
        )

    mutation = pipeline.mutate_strategy(
        {"mutation_contract": contract},
        artifact_id="attempt_140",
        validator=validator,
    )

    assert len(client.calls) == 2
    retry_prompt = client.calls[1]["user"]
    assert "NO_EFFECTIVE_MUTATION" in retry_prompt
    assert '"prose_only_change": true' in retry_prompt
    assert mutation["host_effective_delta"]["changed_strategy_fields"] == [
        "primitive",
        "semantic_objective",
    ]
    assert mutation["host_effective_delta"]["effective"] is True


@pytest.mark.parametrize(
    ("context_name", "level", "parent", "child", "expected_field"),
    [
        (
            "corner_failure",
            "strategy_mutation",
            {
                "primitive": "coordinated_linear_transport",
                "semantic_objective": "forward",
                "coordination_mode": "sync",
            },
            {
                "primitive": "coordinated_pivot",
                "semantic_objective": "counterclockwise",
                "coordination_mode": "async",
            },
            "primitive",
        ),
        (
            "formation_mismatch",
            "coordination_mutation",
            {
                "primitive": "coordinated_alignment",
                "semantic_objective": "longitudinal",
                "coordination_mode": "async",
                "team_assignment": {
                    "front": {"role": "reference"},
                    "back": {"role": "corrector"},
                },
            },
            {
                "primitive": "coordinated_alignment",
                "semantic_objective": "longitudinal",
                "coordination_mode": "async",
                "team_assignment": {
                    "front": {"role": "corrector"},
                    "back": {"role": "reference"},
                },
            },
            "roles",
        ),
        (
            "route_regression",
            "strategy_mutation",
            {
                "primitive": "coordinated_linear_transport",
                "semantic_objective": "right",
                "coordination_mode": "sync",
            },
            {
                "primitive": "coordinated_linear_transport",
                "semantic_objective": "left",
                "coordination_mode": "sync",
            },
            "semantic_objective",
        ),
        (
            "target_alignment_failure",
            "strategy_mutation",
            {
                "primitive": "coordinated_pivot",
                "semantic_objective": "counterclockwise",
                "coordination_mode": "async",
            },
            {
                "primitive": "coordinated_alignment",
                "semantic_objective": "yaw",
                "coordination_mode": "async",
            },
            "primitive",
        ),
    ],
)
def test_host_delta_covers_saved_failure_context_classes(
    context_name,
    level,
    parent,
    child,
    expected_field,
):
    candidate = {
        "mutation_level": level,
        "child_strategy": child,
        "diagnostic_context": context_name,
    }

    delta = host_effective_mutation_delta(
        candidate,
        parent_strategy=parent,
    )

    if expected_field == "roles":
        assert delta["roles_changed"] is True
    else:
        assert expected_field in delta["changed_strategy_fields"]
    assert delta["effective"] is True


def test_host_delta_verifies_actual_backend_parameter_change_for_support_loss():
    parent = {
        "primitive": "coordinated_pivot",
        "semantic_objective": "counterclockwise",
        "coordination_mode": "async",
    }
    candidate = {
        "mutation_level": "parameter_tuning",
        "child_strategy": dict(parent),
        "changed_dimensions": ["angle_deg"],
    }

    delta = host_effective_mutation_delta(
        candidate,
        parent_strategy=parent,
        parent_effective_parameters={
            "angle_deg": {"front": 30.0, "back": 30.0}
        },
        child_effective_parameters={
            "angle_deg": {"front": 15.0, "back": 15.0}
        },
    )

    assert delta["parameter_validation"] == "complete"
    assert delta["changed_parameter_fields"] == ["angle_deg"]
    assert delta["effective"] is True


def test_fixed_sync_primitive_removes_coordination_level_from_shared_mask():
    contract = build_mutation_contract(
        {
            "primitive": "coordinated_linear_transport",
            "semantic_objective": "forward",
            "coordination_mode": "sync",
        }
    )

    assert contract["available_levels"] == [
        "strategy_mutation",
        "parameter_tuning",
    ]
    schema = role_output_schema(
        "evolution_manager",
        {"mutation_contract": contract},
    )
    assert schema["properties"]["mutation_level"]["enum"] == [
        "strategy_mutation",
        "parameter_tuning",
    ]


def test_provider_schema_encodes_nested_role_contracts():
    task_schema = role_output_schema("task_orchestrator", {})
    phase_schema = task_schema["properties"]["global_plan"]["items"]
    subtask_schema = task_schema["properties"]["current_subtask"]
    visual_schema = role_output_schema("vlm_failure_analyzer", {})

    assert set(phase_schema["required"]) == {
        "phase_id",
        "abstract_goal",
        "completion_evidence",
        "status",
    }
    assert "strategy" in subtask_schema["required"]
    assert "setup_only" in subtask_schema["required"]
    assert "team_assignment" in subtask_schema["required"]
    assert set(
        visual_schema["properties"]["failure_analysis"]["required"]
    ) == {
        "outcome",
        "primary_failure",
        "causal_chain",
        "physical_evidence",
        "visual_evidence",
        "strategy_defect",
        "coordination_defect",
        "magnitude_defect",
        "reusable_context",
        "confidence",
    }
    validate_strict_json_schema(task_schema)
    validate_strict_json_schema(visual_schema)
    assert visual_schema["properties"][
        "team_heading_relative_to_visible_free_space"
    ]["enum"] == [
        "aligned_with_open_passage",
        "toward_wall_or_blocked_space",
        "crosswise_to_passage",
        "mixed",
        "uncertain",
    ]
    assert visual_schema["properties"]["failure_analysis"]["properties"][
        "outcome"
    ]["enum"] == ["accepted", "rejected"]


def test_provider_fatal_escapes_role_validation_retries(tmp_path):
    class FailingProvider:
        model_name = DEFAULT_MULTIMODAL_MODEL
        call_records: list[dict] = []

        @staticmethod
        def generate(*, system: str, user: str) -> str:
            raise ProviderFatalError("HTTP 429", http_status=429)

    pipeline = ModelRolePipeline(client=FailingProvider(), output_dir=tmp_path)

    with pytest.raises(ProviderFatalError, match="HTTP 429"):
        pipeline.decide_mutation({}, artifact_id="attempt_10")

    assert len(pipeline.records) == 1
    assert pipeline.records[0]["valid"] is False


def test_unexpected_client_runtime_error_escapes_role_retries(tmp_path):
    class BrokenClient:
        model_name = DEFAULT_MULTIMODAL_MODEL
        call_records: list[dict] = []
        calls = 0

        @classmethod
        def generate(cls, *, system: str, user: str) -> str:
            cls.calls += 1
            raise RuntimeError("unexpected client invariant failure")

    pipeline = ModelRolePipeline(client=BrokenClient(), output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="unexpected client invariant failure"):
        pipeline.decide_mutation({}, artifact_id="attempt_11")

    assert BrokenClient.calls == 1
    assert len(pipeline.records) == 1
    assert pipeline.records[0]["valid"] is False
