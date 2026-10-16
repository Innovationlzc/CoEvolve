from __future__ import annotations

import json

from praxis_coop.cap_policy.generation_recovery import (
    GenerationStallTracker,
    MUTATION_SCOPE_INFEASIBLE,
    NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL,
)
from praxis_coop.cap_policy.model_agents import ModelCollaborativeEvolutionAgent


class _UnusedClient:
    def generate(self, *, system: str, user: str) -> str:
        raise AssertionError("this regression does not call a model")


def test_generation_stalls_stop_on_third_cycle_at_same_checkpoint(tmp_path):
    tracker = GenerationStallTracker(tmp_path)

    events = [
        tracker.record(
            checkpoint_id="accepted_step_04",
            generation_cycle=index,
            role="coder",
            artifact_id="attempt_05",
            error="invalid role response",
            response_count=5,
        )
        for index in (1, 2, 3)
    ]

    assert [item["system_stall"] for item in events] == [False, False, True]
    assert all(item["physical_dispatch"] is False for item in events)
    assert all(item["checkpoint_change"] is False for item in events)
    persisted = json.loads((tmp_path / "generation_stalls.json").read_text())
    assert len(persisted["events"]) == 3


def test_physical_transaction_resets_generation_stall_counter(tmp_path):
    tracker = GenerationStallTracker(tmp_path)
    tracker.record(
        checkpoint_id="accepted_step_04",
        generation_cycle=1,
        role="coder",
        artifact_id="attempt_05",
        error="invalid",
        response_count=5,
    )

    tracker.reset_after_physical_transaction()
    event = tracker.record(
        checkpoint_id="accepted_step_04",
        generation_cycle=1,
        role="coder",
        artifact_id="attempt_06",
        error="invalid",
        response_count=5,
    )

    assert event["consecutive_stalls_at_checkpoint"] == 1
    assert event["system_stall"] is False


def test_successful_role_cycle_resets_generation_stall_counter(tmp_path):
    tracker = GenerationStallTracker(tmp_path)
    tracker.record(
        checkpoint_id="accepted_step_04",
        generation_cycle=1,
        role="vlm_failure_analyzer",
        artifact_id="attempt_06",
        error="invalid",
        response_count=5,
    )

    tracker.reset_after_successful_role_cycle()
    event = tracker.record(
        checkpoint_id="accepted_step_04",
        generation_cycle=1,
        role="coder",
        artifact_id="attempt_07",
        error="invalid",
        response_count=5,
    )

    assert event["consecutive_stalls_at_checkpoint"] == 1


def test_mutation_scope_infeasible_does_not_consume_system_stall_budget(tmp_path):
    tracker = GenerationStallTracker(tmp_path)
    first = tracker.record(
        checkpoint_id="accepted_step_23",
        generation_cycle=1,
        role="evolution_manager",
        artifact_id="attempt_36",
        error="selected coordination mutation has no contract-valid child",
        response_count=1,
        cycle_outcome=MUTATION_SCOPE_INFEASIBLE,
        selected_mutation_level="coordination_mutation",
    )
    second = tracker.record(
        checkpoint_id="accepted_step_23",
        generation_cycle=2,
        role="coder",
        artifact_id="attempt_36",
        error="invalid role response",
        response_count=5,
    )

    assert first["counts_as_model_stall"] is False
    assert first["consecutive_stalls_at_checkpoint"] == 0
    assert first["system_stall"] is False
    assert second["counts_as_model_stall"] is True
    assert second["consecutive_stalls_at_checkpoint"] == 1
    assert second["system_stall"] is False
    assert "backward" not in first["error"]


def test_infeasible_mutation_child_reselects_manager_level_next_cycle(tmp_path):
    agent = ModelCollaborativeEvolutionAgent(
        output_dir=tmp_path,
        client=_UnusedClient(),
    )
    agent._manager_decisions[21] = {"mutation_level": "strategy_mutation"}

    agent.reset_generation_cycle(
        21,
        stall_event={
            "cycle_outcome": NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL,
            "checkpoint_id": "accepted_step_21",
            "generation_cycle": 1,
            "role": "mutation_operator",
            "artifact_id": "attempt_21",
            "selected_mutation_level": "strategy_mutation",
            "response_count": 5,
            "physical_dispatch": False,
            "validation_errors": ["no executable child"],
            "validation_error_kinds": ["candidate_feasibility"],
        },
    )

    assert 21 not in agent._manager_decisions
    feedback = json.loads(
        (tmp_path / "generation_cycle_feedback_attempt_21.json").read_text()
    )
    assert feedback["outcomes"][0]["cycle_outcome"] == (
        NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL
    )
    assert feedback["outcomes"][0]["selected_mutation_level"] == (
        "strategy_mutation"
    )
    assert feedback["outcomes"][0]["minimum_mutation_rank_after"] == 2
    assert feedback["outcomes"][0]["minimum_mutation_level_after"] == (
        "strategy_mutation"
    )


def test_coordination_semantic_failure_monotonically_escalates_to_strategy(tmp_path):
    agent = ModelCollaborativeEvolutionAgent(
        output_dir=tmp_path,
        client=_UnusedClient(),
    )
    agent._manager_decisions[13] = {"mutation_level": "coordination_mutation"}

    agent.reset_generation_cycle(
        13,
        stall_event={
            "cycle_outcome": NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL,
            "checkpoint_id": "accepted_step_11",
            "generation_cycle": 1,
            "role": "task_orchestrator",
            "artifact_id": "attempt_14",
            "selected_mutation_level": "coordination_mutation",
            "typed_failure": "SCOPE_MISMATCH",
            "response_count": 2,
            "physical_dispatch": False,
        },
    )

    assert agent._minimum_mutation_rank_by_checkpoint["accepted_step_11"] == 2

    agent.reset_after_physical_transaction(
        checkpoint_id="accepted_step_11",
        policy_attempt=13,
    )
    assert "accepted_step_11" not in agent._minimum_mutation_rank_by_checkpoint


def test_strategy_failure_stays_at_strategy_without_inventing_a_fourth_level(
    tmp_path,
):
    agent = ModelCollaborativeEvolutionAgent(
        output_dir=tmp_path,
        client=_UnusedClient(),
    )
    agent._minimum_mutation_rank_by_checkpoint["accepted_step_11"] = 2
    agent._manager_decisions[13] = {"mutation_level": "strategy_mutation"}

    agent.reset_generation_cycle(
        13,
        stall_event={
            "cycle_outcome": NO_FEASIBLE_CHILD_AT_SELECTED_LEVEL,
            "checkpoint_id": "accepted_step_11",
            "generation_cycle": 2,
            "role": "mutation_operator",
            "artifact_id": "attempt_14",
            "selected_mutation_level": "strategy_mutation",
            "typed_failure": "NO_EFFECTIVE_DELTA",
            "response_count": 2,
            "physical_dispatch": False,
        },
    )

    assert agent._minimum_mutation_rank_by_checkpoint["accepted_step_11"] == 2
    feedback = json.loads(
        (tmp_path / "generation_cycle_feedback_attempt_13.json").read_text()
    )
    assert feedback["outcomes"][0]["minimum_mutation_rank_after"] == 2
    assert feedback["outcomes"][0]["minimum_mutation_level_after"] == (
        "strategy_mutation"
    )
