from coevolve.envs.m2_box_push_proxy import run_m2_proxy_training


def test_m2_proxy_halypo_satisfies_constraint_and_reaches_goal_region():
    summary, history, _, action = run_m2_proxy_training(method="halypo", updates=12, horizon=60, lr=0.02)

    assert len(history) == 12
    assert summary.constraint_violation_rate == 0.0
    assert summary.final_v <= summary.initial_v
    assert action[0] > 0.0
    assert action[2] > 0.0
