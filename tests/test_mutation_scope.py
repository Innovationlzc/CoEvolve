from __future__ import annotations

from pathlib import Path

import pytest

from praxis_coop.cap_policy.contracts import (
    EvaluationReport,
    ExecutionTrace,
    ObservationBundle,
    PolicyScript,
    RobotAssignment,
    RobotPolicyScript,
)
from praxis_coop.cap_policy.model_agents import (
    ModelCollaborativeEvolutionAgent,
    _mutation_lock_local_navigation_contract,
    _preflight_candidate_infeasible,
    _validate_orchestrator_preserves_locked_child,
    _validate_plan_guided_mutation,
)
from praxis_coop.cap_policy.model_role_pipeline import ModelRoleError
from praxis_coop.cap_policy.mutation_contract import (
    MUTATION_SCOPE_FEASIBLE,
    MUTATION_SCOPE_INFEASIBLE as SCOPE_INFEASIBLE,
    MUTATION_SCOPE_UNKNOWN,
    apply_minimum_mutation_rank,
    apply_infeasible_mutation_level,
    build_mutation_contract,
    check_selected_mutation_scope,
    validate_manager_level,
    validate_mutation_transition,
)

from test_model_role_pipeline import FakeSharedQwenVL, _manager


def _pivot_parent(*, front_role: str = "inner", back_role: str = "outer") -> dict:
    return {
        "primitive": "coordinated_pivot",
        "semantic_objective": "clockwise",
        "formation_strategy": "async coordinated_pivot",
        "coordination_mode": "async",
        "synchronization_intent": "continuous temporary role coordination",
        "team_assignment": {
            "front": {"role": front_role, "objective": "pivot clockwise"},
            "back": {"role": back_role, "objective": "pivot clockwise"},
        },
    }


def _fixed_clockwise_roles() -> dict:
    return {
        "role_assignment_by_direction": {
            "clockwise": {"front": "inner", "back": "outer"},
            "counterclockwise": {"front": "outer", "back": "inner"},
        }
    }


def test_coordination_scope_is_infeasible_when_mode_and_roles_are_fixed():
    scope = check_selected_mutation_scope(
        "coordination_mutation",
        parent_strategy=_pivot_parent(),
        host_pivot_role_contract=_fixed_clockwise_roles(),
    )
    assert scope["status"] == SCOPE_INFEASIBLE
    assert scope["enumerates_actions"] is False
    assert "strategy" not in scope["reason"].lower()
    assert "backward" not in scope["reason"].lower()


def test_coordination_scope_is_feasible_when_a_valid_role_swap_exists():
    scope = check_selected_mutation_scope(
        "coordination_mutation",
        parent_strategy=_pivot_parent(front_role="outer", back_role="inner"),
        host_pivot_role_contract=_fixed_clockwise_roles(),
    )
    assert scope["status"] == MUTATION_SCOPE_FEASIBLE


def test_parameter_and_strategy_scopes_are_not_proven_empty():
    parent = _pivot_parent()
    parameter = check_selected_mutation_scope(
        "parameter_tuning",
        parent_strategy=parent,
    )
    strategy = check_selected_mutation_scope(
        "strategy_mutation",
        parent_strategy=parent,
    )
    assert parameter["status"] == MUTATION_SCOPE_UNKNOWN
    assert strategy["status"] == MUTATION_SCOPE_UNKNOWN


def test_infeasible_level_is_unavailable_without_listing_other_actions():
    contract = build_mutation_contract(_pivot_parent())
    apply_infeasible_mutation_level(contract, "coordination_mutation")
    with pytest.raises(
        ValueError,
        match="no contract-valid effective child",
    ) as caught:
        validate_manager_level(
            {"mutation_level": "coordination_mutation"},
            contract,
        )
    assert "backward" not in str(caught.value)
    assert "strategy_mutation" not in str(caught.value)


def test_minimum_strategy_rank_removes_parameter_and_coordination_levels():
    contract = build_mutation_contract(_pivot_parent())
    apply_minimum_mutation_rank(contract, 2)

    assert contract["minimum_mutation_level"] == "strategy_mutation"
    assert contract["available_levels"] == ["strategy_mutation"]
    with pytest.raises(ValueError, match="unavailable"):
        validate_manager_level(
            {"mutation_level": "coordination_mutation"},
            contract,
        )


def test_orchestrator_must_preserve_host_locked_executable_child():
    locked = _pivot_parent(front_role="outer", back_role="inner")
    plan = {
        "current_subtask": {
            "strategy": {
                key: value
                for key, value in locked.items()
                if key != "team_assignment"
            },
            "team_assignment": locked["team_assignment"],
        }
    }
    _validate_orchestrator_preserves_locked_child(plan, locked_child=locked)

    plan["current_subtask"]["strategy"]["semantic_objective"] = "counterclockwise"
    with pytest.raises(ModelRoleError, match="ORCHESTRATOR_CHILD_MISMATCH"):
        _validate_orchestrator_preserves_locked_child(plan, locked_child=locked)


def _active_local_turn_runtime() -> dict:
    return {
        "checkpoint": {"checkpoint_id": "accepted_step_08"},
        "payload": {"active": True, "x": 2.1, "y": 2.3},
        "robots": {
            "front": {"x": 1.6, "y": 1.8, "yaw_deg": 17.0},
            "back": {"x": 2.2, "y": 2.8, "yaw_deg": 17.0},
        },
        "turn_episode": {
            "schema_version": 3,
            "active": True,
            "status": "local_execution_repair_required",
            "purpose": "local_navigation_turn",
            "reference": "visible_connected_free_space",
            "reference_geometry": "world_direction",
            "world_direction_xy": [0.0, 1.0],
            "direction": "counterclockwise",
            "roles": {"front": "inner", "back": "outer"},
            "primitive": "coordinated_pivot",
            "switch_allowed": False,
            "translation_pending": False,
            "route_observation_captured_at_unix": 123.0,
        },
    }


def _formation_parent_plan() -> dict:
    return {
        "current_subtask": {
            "navigation": {
                "reference": "formation_only",
                "world_direction_xy": [0.0, 0.0],
            },
            "strategy": {
                "primitive": "coordinated_alignment",
                "semantic_objective": "longitudinal",
                "coordination_mode": "async",
            },
            "team_assignment": {
                "front": {"role": "reference"},
                "back": {"role": "corrector"},
            },
        }
    }


def test_mutation_lock_recovers_active_turn_navigation_over_formation_parent():
    contract = _mutation_lock_local_navigation_contract(
        _formation_parent_plan(),
        _active_local_turn_runtime(),
    )

    assert contract["mutation_lock_required"] is True
    assert contract["mutation_lock_source"] == "trusted_runtime_turn_episode"
    assert contract["reference"] == "visible_connected_free_space"
    assert contract["goal_world_direction_xy"] == [0.0, 1.0]
    assert contract["scope_checkpoint_id"] == "accepted_step_08"
    assert contract["preferred_translation_direction"] == "left"
    assert contract["required_turn_direction"] == "counterclockwise"


def test_mutation_lock_keeps_legacy_route_turn_authoritative_without_world_reference():
    runtime = _active_local_turn_runtime()
    runtime["turn_episode"].update({"purpose": "route_turn"})
    runtime["turn_episode"].pop("reference")
    runtime["turn_episode"].pop("reference_geometry")
    runtime["turn_episode"].pop("world_direction_xy")

    contract = _mutation_lock_local_navigation_contract(
        _formation_parent_plan(),
        runtime,
    )

    assert contract["mutation_lock_required"] is True
    assert contract["mutation_lock_source"] == "trusted_runtime_turn_episode"


def test_single_writer_rejects_impossible_child_before_host_lock():
    runtime = _active_local_turn_runtime()
    parent = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "formation_strategy": "repair loaded formation",
        "coordination_mode": "async",
        "synchronization_intent": "front reference and back corrector",
        "team_assignment": {
            "front": {"role": "reference", "objective": "hold"},
            "back": {"role": "corrector", "objective": "repair"},
        },
    }
    child = {
        "primitive": "coordinated_linear_transport",
        "semantic_objective": "forward",
        "formation_strategy": "matched loaded transport",
        "coordination_mode": "sync",
        "synchronization_intent": "move together",
        "team_assignment": {
            "front": {"role": "follower", "objective": "support"},
            "back": {"role": "reference", "objective": "support"},
        },
    }
    contract = _mutation_lock_local_navigation_contract(
        _formation_parent_plan(),
        runtime,
    )

    with pytest.raises(ModelRoleError, match="declared visible-passage direction"):
        _validate_plan_guided_mutation(
            {"mutation_level": "strategy_mutation", "child_strategy": child},
            manager_decision={"mutation_level": "strategy_mutation"},
            parent_strategy=parent,
            allowed_sdk_methods={
                "coordinated_alignment",
                "coordinated_linear_transport",
                "coordinated_pivot",
            },
            runtime_state=runtime,
            local_navigation_contract=contract,
            model_owns_route=True,
            mutation_contract=build_mutation_contract(parent),
        )

    pivot = {
        **child,
        "primitive": "coordinated_pivot",
        "semantic_objective": "counterclockwise",
        "coordination_mode": "async",
        "team_assignment": {
            "front": {"role": "inner", "objective": "hold pivot"},
            "back": {"role": "outer", "objective": "swing"},
        },
    }
    _validate_plan_guided_mutation(
        {"mutation_level": "strategy_mutation", "child_strategy": pivot},
        manager_decision={"mutation_level": "strategy_mutation"},
        parent_strategy=parent,
        allowed_sdk_methods={
            "coordinated_alignment",
            "coordinated_linear_transport",
            "coordinated_pivot",
        },
        runtime_state=runtime,
        local_navigation_contract=contract,
        model_owns_route=True,
        mutation_contract=build_mutation_contract(parent),
    )


def test_feasible_coordination_mutation_still_cannot_change_primitive():
    parent = {
        "primitive": "coordinated_alignment",
        "semantic_objective": "longitudinal",
        "formation_strategy": "async coordinated_alignment",
        "coordination_mode": "async",
        "synchronization_intent": "continuous temporary role coordination",
        "team_assignment": {
            "front": {"role": "corrector", "objective": "close the gap"},
            "back": {"role": "anchor", "objective": "hold"},
        },
    }
    child = dict(parent)
    child["primitive"] = "coordinated_linear_transport"
    child["semantic_objective"] = "backward"
    with pytest.raises(ValueError, match="coordination_mutation mutable mask"):
        validate_mutation_transition(
            {
                "mutation_level": "coordination_mutation",
                "child_strategy": child,
            },
            parent_strategy=parent,
            contract=build_mutation_contract(parent),
        )


def test_turn_not_ready_is_preflight_candidate_infeasible():
    report = EvaluationReport(
        success=False,
        failure_type="turn_not_ready",
        evidence={
            "turn_episode_preflight": {
                "physically_dispatched": False,
                "error": "PREFLIGHT_TURN_CLEARANCE_INSUFFICIENT: swept clearance",
            }
        },
    )
    assert _preflight_candidate_infeasible(report) is True
    assert (
        _preflight_candidate_infeasible(
            EvaluationReport(success=False, failure_type="formation_desync")
        )
        is False
    )


def _pivot_policy() -> PolicyScript:
    def code(role: str) -> str:
        return (
            "def policy(api):\n"
            "    api.coordinated_pivot(turn_direction='clockwise', "
            "angle_deg=10.0, duration_s=8.0, "
            f"role='{role}', swing_radius_m=0.8)\n"
            "    api.stop_all()\n"
        )

    return PolicyScript(
        script_id="attempt_35",
        attempt=35,
        code=code("inner"),
        assignments=(
            RobotAssignment(
                robot_id="front",
                role="inner",
                objective="hold inner",
                synchronization_group="turn_team",
            ),
            RobotAssignment(
                robot_id="back",
                role="outer",
                objective="swing outer",
                synchronization_group="turn_team",
            ),
        ),
        robot_scripts=(
            RobotPolicyScript(
                "front",
                "front_attempt_35",
                code("inner"),
                "turn_team",
                coordination_mode="async",
            ),
            RobotPolicyScript(
                "back",
                "back_attempt_35",
                code("outer"),
                "turn_team",
                coordination_mode="async",
            ),
        ),
    )


def test_preflight_turn_not_ready_skips_evolution_manager(tmp_path: Path):
    from praxis_coop.cap_policy.model_role_pipeline import ModelRolePipeline

    client = FakeSharedQwenVL([])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        role_pipeline=pipeline,
        formal_eval=True,
        coordination_layer_enabled=True,
    )
    decision = agent.decide_next_step(
        _pivot_policy(),
        EvaluationReport(success=False, failure_type="turn_not_ready"),
        ObservationBundle(
            runtime_state={
                "robots": {
                    "front": {"x": 0.8, "y": 7.5, "yaw_rad": 1.75},
                    "back": {"x": -0.6, "y": 7.4, "yaw_rad": 1.75},
                },
                "payload": {"x": 0.1, "y": 7.45, "active": True},
            }
        ),
        ExecutionTrace(script_id="attempt_35", attempt=35, status="failed"),
        checkpoint_id="accepted_step_23",
        physically_dispatched=False,
    )
    assert decision.should_evolve is True
    assert client.calls == []
    assert agent._manager_decisions == {}


def test_selected_empty_coordination_scope_reselects_inside_manager_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from praxis_coop.cap_policy.model_role_pipeline import ModelRolePipeline

    client = FakeSharedQwenVL(
        [_manager("coordination_mutation"), _manager("strategy_mutation")]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        role_pipeline=pipeline,
        formal_eval=True,
        coordination_layer_enabled=True,
    )

    observed_scope_contracts = []

    def fake_scope(level, **kwargs):
        observed_scope_contracts.append(kwargs.get("host_pivot_role_contract"))
        if level != "coordination_mutation":
            return {
                "status": MUTATION_SCOPE_UNKNOWN,
                "selected_level": level,
                "reason": "strategy children are not exhaustively enumerable",
                "enumerates_actions": False,
            }
        return {
            "status": SCOPE_INFEASIBLE,
            "selected_level": level,
            "reason": (
                "The selected mutation-level operator has no contract-valid "
                "effective child at the current checkpoint. Reselect a mutation "
                "level based on the current evidence."
            ),
            "enumerates_actions": False,
        }

    monkeypatch.setattr(
        "praxis_coop.cap_policy.model_agents.check_selected_mutation_scope",
        fake_scope,
    )
    decision = agent.decide_next_step(
        _pivot_policy(),
        EvaluationReport(success=False, failure_type="payload_drop"),
        ObservationBundle(
            runtime_state={
                "robots": {
                    "front": {"x": 0.8, "y": 7.5, "yaw_rad": 1.75},
                    "back": {"x": -0.6, "y": 7.4, "yaw_rad": 1.75},
                },
                "payload": {"x": 0.1, "y": 7.45, "active": True},
            }
        ),
        ExecutionTrace(
            script_id="attempt_35",
            attempt=35,
            status="script_error",
        ),
        checkpoint_id="accepted_step_23",
        physically_dispatched=True,
    )
    assert decision.should_evolve is True
    assert agent._manager_decisions[35]["mutation_level"] == "strategy_mutation"
    assert len(client.calls) == 2
    assert "You are the Evolution Manager" in client.calls[0]["system"]
    assert observed_scope_contracts == [{}, {}]
    assert "coordination_mutation" in agent._infeasible_mutation_levels["accepted_step_23"]


def test_manager_below_minimum_rank_retries_without_generation_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from praxis_coop.cap_policy.model_role_pipeline import ModelRolePipeline

    client = FakeSharedQwenVL(
        [_manager("coordination_mutation"), _manager("strategy_mutation")]
    )
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path,
        client=client,
        role_pipeline=pipeline,
        formal_eval=True,
        coordination_layer_enabled=True,
    )
    agent._minimum_mutation_rank_by_checkpoint["accepted_step_23"] = 2
    monkeypatch.setattr(
        "praxis_coop.cap_policy.model_agents.check_selected_mutation_scope",
        lambda level, **kwargs: {
            "status": MUTATION_SCOPE_UNKNOWN,
            "selected_level": level,
            "reason": "strategy children are not exhaustively enumerable",
            "enumerates_actions": False,
        },
    )

    decision = agent.decide_next_step(
        _pivot_policy(),
        EvaluationReport(success=False, failure_type="payload_drop"),
        ObservationBundle(
            runtime_state={
                "checkpoint": {"checkpoint_id": "accepted_step_23"},
                "payload": {"x": 0.1, "y": 7.45, "active": True},
            }
        ),
        ExecutionTrace(script_id="attempt_35", attempt=35, status="script_error"),
        checkpoint_id="accepted_step_23",
        physically_dispatched=False,
    )

    assert decision.should_evolve is True
    assert agent._manager_decisions[35]["mutation_level"] == "strategy_mutation"
    assert [record["valid"] for record in pipeline.records] == [False, True]
    assert not (tmp_path / "generation_cycle_feedback_attempt_35.json").exists()
