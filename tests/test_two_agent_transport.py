from __future__ import annotations

from coevolve.envs.two_agent_transport import run_transport_training


def test_m1_halypo_reduces_gap_and_satisfies_constraint() -> None:
    independent_summary, _ = run_transport_training(method="independent", updates=60)
    halypo_summary, _ = run_transport_training(method="halypo", updates=60)

    assert halypo_summary.final_v < halypo_summary.initial_v
    assert halypo_summary.final_v < independent_summary.final_v
    assert halypo_summary.final_team_return > halypo_summary.initial_team_return
    assert halypo_summary.constraint_violation_rate == 0.0
    assert independent_summary.constraint_violation_rate > halypo_summary.constraint_violation_rate
