"""Offline regressions for the recorded Table 4 accepted-rewind deadlocks."""

from dataclasses import replace

import pytest

from praxis_coop.cap_policy import (
    EvaluationReport,
    ExecutionTrace,
    ModelCollaborativeEvolutionAgent,
    ModelRolePipeline,
    ObservationBundle,
)
from praxis_coop.cap_policy.model_agents import (
    ModelAgentError,
    _execution_strategy_from_task_plan,
    _final_approach_transport_envelope,
    _generate_plan_guided_team_policy,
    _loaded_linear_team_transaction_signature,
    _release_preflight_planning_context,
    _validate_failure_level_repair,
    _validate_final_approach_strategy_availability,
    _validate_final_approach_transport_policy,
    _validate_policy_within_host_wall_clearance,
)
from praxis_coop.cap_policy.model_role_pipeline import ModelRoleError
from test_model_role_pipeline import (
    FakeSharedQwenVL,
    _policy,
    _sync_linear_team_policy_payload,
    _task_for_plan,
    _task_plan,
)
from test_release_preflight_evidence import _state as _release_state


def _envelope(*, clearance_m=0.44, direction="forward", **kwargs):
    plan = _task_plan()
    plan["current_phase_id"] = "phase_05_final_approach"
    strategy = _execution_strategy_from_task_plan(plan)
    strategy["semantic_objective"] = direction
    return _final_approach_transport_envelope(
        {"payload": {"x": 0.0, "y": 0.0},
         "target": {"center_x_m": 0.7, "center_y_m": 0.0}},
        task_plan=plan,
        execution_strategy=strategy,
        host_wall_clearance={"translation_candidates": {
            "forward": {"clearance_m": clearance_m},
            "left": {"clearance_m": 0.9},
        }},
        **kwargs,
    )


def test_accepted_rewind_does_not_require_coder_to_change_committed_strategy(tmp_path):
    """Exercise the actual Orchestrator -> Coder path, no physics or API."""
    payload = _sync_linear_team_policy_payload()
    for script in payload["robot_scripts"].values():
        script["code"] = script["code"].replace("distance_m=1.0", "distance_m=0.5")
    client = FakeSharedQwenVL([_task_plan(), payload])
    pipeline = ModelRolePipeline(client=client, output_dir=tmp_path)
    pipeline._current_task_plan = _task_plan()
    agent = ModelCollaborativeEvolutionAgent(
        tmp_path, client=client,
        task=_task_for_plan(_task_plan(), route_discovery=True),
        route_discovery=True, coordination_layer_enabled=True,
        role_pipeline=pipeline,
    )
    observation = ObservationBundle(runtime_state={
        "checkpoint": {"checkpoint_id": "accepted_step_13"},
        "payload": {"present": True, "active": True,
                    "contacts": {"support_robots": ["front", "back"],
                                 "floor": False, "walls": []}},
    })
    report = EvaluationReport(success=True, failure_type="none",
                              metrics={"task_complete": 0.0})
    parent = _policy()
    decision = agent.decide_next_step(
        parent, report, observation,
        ExecutionTrace(script_id=parent.script_id, attempt=1, status="success"),
        checkpoint_id="accepted_step_13", physically_dispatched=True,
    )
    rewind_decision = replace(decision, repair_level="strategy",
                              reason="descendant generation dead end")
    evolved, _ = agent.evolve(parent, report, observation,
                              controller_decision=rewind_decision)
    assert "distance_m=0.5" in evolved.robot_scripts[0].code
    assert [record["role"] for record in pipeline.records] == [
        "task_orchestrator", "coder",
    ]
    assert all(record["valid"] for record in pipeline.records)


def test_physical_strategy_failure_still_rejects_parameter_only_repair():
    with pytest.raises(ModelAgentError, match="changing only numerical parameters"):
        _validate_failure_level_repair(
            _policy(distance_m=1.0), _policy(distance_m=0.5),
            failure_level="strategy", failure_type="route_regression",
            blocked_strategy_signatures=set(),
        )


def test_target_distance_and_wall_clearance_have_nonempty_common_tier():
    envelope = _envelope()
    assert envelope["required_distance_m"] == 0.25
    assert envelope["allowed_distance_m"] == [0.25]
    assert envelope["clearance_excluded_distance_m"] == [0.5]
    assert envelope["required_speed_mps"] == 0.5
    assert envelope["required_ramp_s"] == 1.5
    policy = _policy(distance_m=0.25)
    _validate_final_approach_transport_policy(policy, envelope=envelope)
    _validate_policy_within_host_wall_clearance(
        policy, clearance={"translation_candidates": {"forward": {"clearance_m": 0.44}}}
    )
    with pytest.raises(ModelAgentError, match="authored-wall clearance"):
        _validate_policy_within_host_wall_clearance(
            _policy(distance_m=0.5),
            clearance={"translation_candidates": {"forward": {"clearance_m": 0.44}}},
        )


def test_clearance_belongs_to_committed_direction_not_other_candidates():
    assert _envelope(direction="left")["required_distance_m"] == 0.5


@pytest.mark.parametrize("clearance_m", [0.24, 0.0, -0.1])
def test_empty_distance_intersection_is_rejected_before_coder(clearance_m):
    envelope = _envelope(clearance_m=clearance_m)
    assert envelope["exhausted"] is True
    assert "required_distance_m" not in envelope
    with pytest.raises(ModelRoleError, match="unavailable"):
        _validate_final_approach_strategy_availability(
            method="coordinated_linear_transport", envelope=envelope,
        )


def test_rejected_short_ast_does_not_reopen_long_unsafe_distance():
    signature = _loaded_linear_team_transaction_signature(direction="forward", distance_m=0.25)
    envelope = _envelope(forbidden_transaction_signatures={signature})
    assert envelope["exhausted"] is True


def test_exact_sync_probe_cannot_be_silently_shortened_to_fit_wall():
    envelope = _envelope(formation_sync_repair_gate={
        "active": True,
        "probe_contract": {"semantic_objective": "forward", "required_distance_m": 0.5},
    })
    assert envelope["exhausted"] is True
    assert envelope["allowed_distance_m"] == []


def test_coder_can_instantiate_short_clearance_safe_final_approach(tmp_path):
    payload = _sync_linear_team_policy_payload()
    for script in payload["robot_scripts"].values():
        script["code"] = script["code"].replace("distance_m=1.0", "distance_m=0.25")
    pipeline = ModelRolePipeline(client=FakeSharedQwenVL([payload]), output_dir=tmp_path)
    plan = _task_plan()
    _, policy = _generate_plan_guided_team_policy(
        role_pipeline=pipeline, task_plan=plan,
        execution_strategy=_execution_strategy_from_task_plan(plan),
        generation_context={
            "final_approach_transport_envelope": _envelope(),
            "host_authored_wall_clearance": {"translation_candidates": {
                "forward": {"clearance_m": 0.44},
            }},
        },
        attempt=44, fallback_id="attempt_44", parent_script_id="attempt_43",
        require_team_plan=True, expected_method="",
        forbidden_transaction_signatures=(), current_policy=None,
        repair_level="", repair_failure_type="",
    )
    assert "distance_m=0.25" in policy.robot_scripts[0].code
    assert len(pipeline.records) == 1
    assert pipeline.records[0]["valid"] is True


def test_release_evidence_does_not_leak_target_into_early_route_discovery():
    assert _release_preflight_planning_context(
        _release_state(), route_discovery=True,
        lifecycle={"lifecycle_state": "carry"}, final_approach_envelope={},
    ) == {}


@pytest.mark.parametrize("lifecycle,envelope", [
    ({"lifecycle_state": "place"}, {}),
    ({"lifecycle_state": "release"}, {}),
    ({"lifecycle_state": "carry"}, {"active": True}),
])
def test_local_release_evidence_uses_full_runtime_and_never_grants_dispatch(lifecycle, envelope):
    evidence = _release_preflight_planning_context(
        _release_state(), route_discovery=True,
        lifecycle=lifecycle, final_approach_envelope=envelope,
    )
    assert evidence["available"] is True
    assert evidence["target_minus_payload_world_xy_m"] == [-0.25, 0.0]
    assert evidence["dispatch_gate_still_required"] is True
    assert evidence["snapshot_stability_certified"] is False
