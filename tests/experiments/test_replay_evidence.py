from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from coevolve_experiments.reporting.replay_evidence import (
    ACCEPTED_MANIFEST_PIN,
    ReplayHistoryMode,
    StrictReplayEvidenceError,
    execute_strict_replay,
    load_strict_replay_manifest,
    prepare_strict_replay_manifest,
    validate_and_publish_strict_replay_result,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _source(tmp_path: Path, *, steps: int = 2) -> tuple[Path, Path]:
    run = tmp_path / "run"
    package = run / "all_scripts"
    accepted_steps = []
    trajectory_steps = []
    checkpoint = "accepted_step_00"
    for index in range(1, steps + 1):
        next_checkpoint = f"accepted_step_{index:02d}"
        scripts = {
            robot: {
                "robot_id": robot,
                "code": (
                    "def policy(api):\n"
                    "    api.move_forward(distance_m=0.1)\n"
                    "    api.stop_all()"
                ),
            }
            for robot in ("front", "back")
        }
        source = {
            "index": index,
            "attempt": index + 2,
            "script_id": f"move_{index}",
            "checkpoint_before": checkpoint,
            "checkpoint_after": next_checkpoint,
            "transition_outcome": (
                "TASK_COMPLETE" if index == steps else "ACCEPT_PROGRESS"
            ),
            "robot_scripts": scripts,
        }
        accepted_steps.append(source)
        trajectory_steps.append(
            {
                **source,
                "sequence_index": index,
                "robot_scripts": {
                    robot: {**script, "code": script["code"] + "\n"}
                    for robot, script in scripts.items()
                },
            }
        )
        checkpoint = next_checkpoint
    accepted = {
        "kind": "accepted_checkpoint_episode",
        "schema_version": 1,
        "initial_checkpoint": "accepted_step_00",
        "latest_checkpoint": checkpoint,
        "steps": accepted_steps,
    }
    accepted_path = run / "memory" / "accepted_episode_manifest.json"
    _write(accepted_path, accepted)
    accepted_sha = hashlib.sha256(accepted_path.read_bytes()).hexdigest()
    _write(
        run / "run_manifest.json",
        {
            "status": "SUCCESS",
            "run_id": "paper-run-200",
            "result_artifact_hashes": {ACCEPTED_MANIFEST_PIN: accepted_sha},
        },
    )
    _write(
        package / "trajectory_manifest.json",
        {
            "kind": "trusted_accepted_episode_export",
            "schema_version": 1,
            "accepted_action_count": steps,
            "steps": trajectory_steps,
        },
    )
    return run, package


def _result_files(
    replay: Path,
    *,
    steps: int,
    planner_reset: bool,
) -> None:
    _write(
        replay / "replay_config.json",
        {
            "from_step": 1,
            "through_step": steps,
            "steps_requested": steps,
            "planner_reset_each_step": planner_reset,
        },
    )
    results = [
        {
            "sequence_index": index,
            "physical_success": True,
            "accepted_endpoint_deviation": {
                "payload_xy_error_m": 0.01 * index,
                "payload_z_error_m": 0.02 * index,
            },
        }
        for index in range(1, steps + 1)
    ]
    _write(
        replay / "replay_summary.json",
        {
            "status": "completed",
            "completed_steps": steps,
            "final_physical_gate": {"success": True},
            "results": results,
        },
    )


def _argv(binding) -> list[str]:
    flag = (
        "--planner-reset-each-step"
        if binding.mode is ReplayHistoryMode.RESET_EACH_CHECKPOINT
        else "--no-planner-reset-each-step"
    )
    return [
        "/test/python",
        str(binding.executor_path),
        "replay",
        "--run-dir",
        str(binding.run_root),
        "--package-dir",
        str(binding.package_root),
        "--execute",
        "--from-step",
        "1",
        "--through-step",
        str(binding.action_count),
        "--gate-timeout-s",
        "10.0",
        flag,
    ]


def test_manifest_pins_success_run_manifest_sequence_and_executor(tmp_path: Path):
    run, package = _source(tmp_path)
    binding = prepare_strict_replay_manifest(
        run,
        package,
        mode=ReplayHistoryMode.RESET_EACH_CHECKPOINT,
    )

    assert binding.action_count == 2
    assert binding.mode is ReplayHistoryMode.RESET_EACH_CHECKPOINT
    assert binding.document["diagnostic_only"] is True
    assert binding.document["executor"]["relative_path"] == (
        "scripts/manage_accepted_episode.py"
    )
    assert load_strict_replay_manifest(
        binding.manifest_path,
        run_root=run,
        package_root=package,
    ).manifest_sha256 == binding.manifest_sha256


def test_manifest_rejects_partial_or_tampered_trajectory(tmp_path: Path):
    run, package = _source(tmp_path)
    trajectory_path = package / "trajectory_manifest.json"
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["steps"].pop()
    _write(trajectory_path, trajectory)
    with pytest.raises(StrictReplayEvidenceError, match="complete accepted sequence"):
        prepare_strict_replay_manifest(
            run,
            package,
            mode=ReplayHistoryMode.CONTINUOUS,
        )

    run, package = _source(tmp_path / "second")
    binding = prepare_strict_replay_manifest(
        run,
        package,
        mode=ReplayHistoryMode.CONTINUOUS,
    )
    trajectory_path = package / "trajectory_manifest.json"
    trajectory = json.loads(trajectory_path.read_text())
    trajectory["steps"][0]["robot_scripts"]["front"]["code"] += " # changed"
    _write(trajectory_path, trajectory)
    with pytest.raises(
        StrictReplayEvidenceError,
        match=r"trajectory (size|hash) changed",
    ):
        load_strict_replay_manifest(
            binding.manifest_path,
            run_root=run,
            package_root=package,
        )


def test_result_derives_safe_prefix_and_maximum_payload_divergence(tmp_path: Path):
    run, package = _source(tmp_path)
    binding = prepare_strict_replay_manifest(
        run,
        package,
        mode=ReplayHistoryMode.CONTINUOUS,
    )
    replay = package / "replays" / "replay_test"
    _result_files(replay, steps=2, planner_reset=False)

    result = validate_and_publish_strict_replay_result(
        binding,
        replay,
        process_returncode=0,
        argv=_argv(binding),
    )

    assert result.full_success is True
    assert result.safe_replay_prefix == result.total_actions == 2
    assert result.failed_step is None
    assert result.maximum_payload_trajectory_divergence_m == pytest.approx(
        math.hypot(0.02, 0.04)
    )
    assert result.result_path.is_file()


def test_result_rejects_mode_mismatch_and_success_after_failure(tmp_path: Path):
    run, package = _source(tmp_path)
    binding = prepare_strict_replay_manifest(
        run,
        package,
        mode=ReplayHistoryMode.RESET_EACH_CHECKPOINT,
    )
    replay = package / "replays" / "replay_test"
    _result_files(replay, steps=2, planner_reset=False)
    with pytest.raises(StrictReplayEvidenceError, match="planner_reset_each_step"):
        validate_and_publish_strict_replay_result(
            binding,
            replay,
            process_returncode=0,
            argv=_argv(binding),
        )

    # Rebuild a valid config but make the trace continue after a failure.
    _result_files(replay, steps=2, planner_reset=True)
    summary_path = replay / "replay_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "failed_and_rolled_back"
    summary["failed_step"] = 1
    summary["completed_steps"] = 0
    summary["results"][0]["physical_success"] = False
    _write(summary_path, summary)
    with pytest.raises(StrictReplayEvidenceError, match="continued after"):
        validate_and_publish_strict_replay_result(
            binding,
            replay,
            process_returncode=2,
            argv=_argv(binding),
        )


def test_execute_accepts_virtualenv_style_python_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run, package = _source(tmp_path)
    binding = prepare_strict_replay_manifest(
        run,
        package,
        mode=ReplayHistoryMode.CONTINUOUS,
    )
    interpreter_link = tmp_path / "python"
    os.symlink(Path(sys.executable).resolve(), interpreter_link)

    def fake_run(argv, **_kwargs):
        assert Path(argv[0]) == Path(sys.executable).resolve()
        replay = package / "replays" / "replay_symlink_interpreter"
        _result_files(replay, steps=2, planner_reset=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "coevolve_experiments.reporting.replay_evidence.subprocess.run",
        fake_run,
    )
    result = execute_strict_replay(
        binding,
        python_executable=interpreter_link,
    )
    assert result.full_success is True
