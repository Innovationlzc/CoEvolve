from __future__ import annotations

import json
from pathlib import Path

from praxis_coop.cap_policy.result_review import build_run_review


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "seed_000"
    attempt = run_dir / "attempt_01"
    attempt.mkdir(parents=True)
    policy = {
        "attempt": 1,
        "script_id": "turn_once",
        "assignments": [
            {
                "robot_id": "front",
                "role": "outer",
                "phase_id": "route_phase_0",
                "coordination_mode": "async",
            },
            {
                "robot_id": "back",
                "role": "inner",
                "phase_id": "route_phase_0",
                "coordination_mode": "async",
            },
        ],
        "robot_scripts": [
            {
                "robot_id": "front",
                "script_id": "front_turn",
                "code": (
                    "def policy(api):\n"
                    "    api.coordinated_pivot("
                    'turn_direction="counterclockwise", angle_deg=15.0)'
                ),
            },
            {
                "robot_id": "back",
                "script_id": "back_turn",
                "code": (
                    "def policy(api):\n"
                    "    api.coordinated_pivot("
                    'turn_direction="counterclockwise", angle_deg=15.0)'
                ),
            },
        ],
    }
    _write_json(attempt / "policy.json", policy)
    _write_json(attempt / "team_plan.json", policy)
    _write_json(
        attempt / "task_distribution.json",
        {
            "coordination_mode": "async",
            "checkpoint": {
                "before": "accepted_step_00",
                "after": "accepted_step_01",
                "rollback_performed": False,
            },
        },
    )
    _write_json(
        attempt / "execution_trace.json",
        {
            "status": "success",
            "skill_calls": [{"status": "success"}],
            "robot_executions": [
                {
                    "robot_id": robot_id,
                    "planned_calls": [
                        {
                            "method": "coordinated_pivot",
                            "parameters": {
                                "turn_direction": "counterclockwise",
                                "angle_deg": 15.0,
                            },
                        }
                    ],
                }
                for robot_id in ("front", "back")
            ],
        },
    )
    _write_json(
        attempt / "evaluation.json",
        {
            "success": True,
            "failure_type": "none",
            "failure_level": "none",
            "diagnosis": "accepted",
            "metrics": {"payload_step_displacement_m": 0.1},
        },
    )
    _write_json(
        attempt / "transition_decision.json",
        {"outcome": "ACCEPT_PROGRESS", "reason": "safe progress"},
    )
    _write_json(
        attempt / "evolution_controller_decision.json",
        {"action": "continue", "repair_level": "none"},
    )
    _write_json(
        attempt / "evolution_record.json",
        {
            "checkpoint_before": "accepted_step_00",
            "checkpoint_after": "accepted_step_01",
            "rollback_performed": False,
            "next_script_id": "next_action",
        },
    )
    (attempt / "front_turn.py").write_text("def policy(api): pass\n")
    (attempt / "skill_calls").mkdir()
    (attempt / "skill_calls" / "stdout.log").write_text("ok\n")

    visual = run_dir / "visual_observations" / "attempt_01_post.png"
    visual.parent.mkdir()
    visual.write_bytes(b"png")
    role_artifact = run_dir / "model_roles" / "coder_attempt_01_front.json"
    _write_json(role_artifact, {"role": "coder", "valid": True})
    _write_json(
        run_dir / "model_roles" / "role_call_index.json",
        {
            "calls": [
                {
                    "role": "coder",
                    "artifact_id": "attempt_01_front",
                    "artifact_path": str(role_artifact),
                    "valid": True,
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "billing_amount_microusd": 50,
                    },
                }
            ]
        },
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            "seed": 0,
            "protocol": {"protocol_id": "test-protocol"},
            "model": {"requested": "test-model"},
            "codebase": {"git_head": "abc123"},
        },
    )
    return run_dir


def test_build_run_review_groups_artifacts_without_moving_raw_files(tmp_path):
    run_dir = _make_run(tmp_path)
    raw_policy = (run_dir / "attempt_01" / "policy.json").read_bytes()

    summary = build_run_review(run_dir)

    review = run_dir / "review"
    attempt_review = review / "attempts" / "attempt_01"
    assert summary["attempt_count"] == 1
    assert summary["accepted_attempt_count"] == 1
    assert summary["model_calls"]["call_count"] == 1
    assert summary["model_calls"]["known_billing_amount_microusd"] == 50
    assert (run_dir / "attempt_01" / "policy.json").read_bytes() == raw_policy
    assert (attempt_review / "01_planning" / "policy.json").is_symlink()
    assert (attempt_review / "02_generated_code" / "front_turn.py").is_symlink()
    assert (attempt_review / "03_execution" / "skill_calls").is_symlink()
    assert (attempt_review / "04_verification" / "evaluation.json").is_symlink()
    assert (
        attempt_review
        / "07_model_calls"
        / "coder"
        / "coder_attempt_01_front.json"
    ).is_symlink()
    assert (attempt_review / "08_visuals" / "attempt_01_post.png").is_symlink()
    assert not list(review.rglob("*")) == []
    assert not [path for path in review.rglob("*") if path.is_symlink() and not path.exists()]

    attempt_summary = json.loads(
        (attempt_review / "summary.json").read_text(encoding="utf-8")
    )
    assert attempt_summary["semantic_objective"] == "counterclockwise"
    assert attempt_summary["checkpoint_after"] == "accepted_step_01"
    assert attempt_summary["transition_outcome"] == "ACCEPT_PROGRESS"
    attempt_readme = (attempt_review / "README.md").read_text(encoding="utf-8")
    assert "$0.000050" in attempt_readme
    assert "unknown" not in attempt_readme.split("## 模型调用", 1)[1].split(
        "##", 1
    )[0]


def test_build_run_review_is_rebuildable_and_removes_stale_derived_files(tmp_path):
    run_dir = _make_run(tmp_path)
    build_run_review(run_dir)
    stale = run_dir / "review" / "stale.txt"
    stale.write_text("derived", encoding="utf-8")

    summary = build_run_review(run_dir, review_dir="review")

    assert summary["attempt_count"] == 1
    assert not stale.exists()
    assert (run_dir / "review" / "README.md").is_file()


def test_build_run_review_rejects_destination_outside_run(tmp_path):
    run_dir = _make_run(tmp_path)

    try:
        build_run_review(run_dir, review_dir=tmp_path / "outside")
    except ValueError as exc:
        assert "must be a child" in str(exc)
    else:
        raise AssertionError("outside review directory should be rejected")
