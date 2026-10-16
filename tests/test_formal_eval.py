import json
import sys
from types import SimpleNamespace

import pytest

import praxis_coop.cap_policy.formal_eval as formal_eval_module
from praxis_coop.cap_policy.contracts import (
    BoxCarryTask,
    EvaluationReport,
    ExecutionTrace,
    PolicyScript,
    RouteProgressGate,
)
from praxis_coop.cap_policy.evolution_controller import EvolutionController
from praxis_coop.cap_policy.formal_eval import (
    FORMAL_BACKBONES,
    FORMAL_EVAL_PROTOCOL_ID,
    FORMAL_MAX_NEW_TOKENS,
    FORMAL_MODEL_ATTEMPT_BUDGET,
    FORMAL_PHYSICAL_BUDGET,
    FORMAL_PROVIDER_NETWORK_RETRIES,
    FORMAL_RUN_COUNT,
    FormalEvalConfigurationError,
    formal_allows_ancestor_rewind,
    formal_memory_paths,
    formal_run_output_dir,
    validate_formal_freeze_commit,
    validate_formal_initial_runtime,
    validate_fresh_formal_run,
    write_formal_run_manifest,
)
from praxis_coop.cap_policy.simulation_diagnostics import _target_metrics
from scripts.run_cap_box_carry_evolution import (
    REPO_ROOT,
    _append_jsonl_event,
    _advance_physical_transaction_commit,
    _all_recovery_levels_deterministically_infeasible,
    _begin_physical_transaction_commit,
    _checkpoint_reference_after_ancestor_rewind,
    _controller_metrics_with_dispatch_journals,
    _finalize_experiment,
    _finalize_physical_transaction_outcome,
    _mark_transaction_journals_terminal_metrics,
    _phase_completion_metrics,
    _post_dispatch_model_terminal_report,
    _run_checkpoint_generation_cycles,
    _terminal_runtime_fault,
    _terminal_reason_for_exception,
    parse_args,
)
from praxis_coop.cap_policy.generation_recovery import (
    GenerationStallTracker,
    HostContractError,
    MUTATION_SCOPE_INFEASIBLE,
    SystemStallError,
)
from praxis_coop.cap_policy.model_role_pipeline import ModelRoleStallError
from praxis_coop.llm import ProviderFatalError
from praxis_coop.llm.billing_ledger import write_api_cost_ledger
from scripts.prepare_formal_evaluation import prepare_run_directory
from scripts.reconcile_api_costs import _refresh_usage_billing_summaries
from scripts.summarize_formal_evaluation import (
    _artifact_protocol_invalid_reasons,
    _billing_artifact_invalid_reasons,
    _is_cost_complete,
    _is_protocol_valid_run,
    _system_stall_evidence_invalid_reasons,
    _wilson_interval,
    summarize_rows,
)


def _seal_complete_billing_fixture(output_dir, usage, metrics=None, manifest=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    _refresh_usage_billing_summaries(usage)
    ledger = write_api_cost_ledger(output_dir, usage, experiment={"seed": 0})
    summary = ledger["summary"]
    report = {
        "status": "complete",
        "remaining": 0,
        "source": "post_terminal_provider_request_reconciliation",
        "clients": [],
        "ledger_billing_complete": True,
        "billing_amount_microusd": summary["billing_amount_microusd"],
        "billing_amount_usd": summary["billing_amount_usd"],
    }
    usage["billing_reconciliation"] = report
    (output_dir / "model_usage.json").write_text(
        json.dumps(usage) + "\n",
        encoding="utf-8",
    )
    (output_dir / "billing_reconciliation.json").write_text(
        json.dumps(report) + "\n",
        encoding="utf-8",
    )
    (output_dir / "billing_post_reconciliation.json").write_text(
        json.dumps(report) + "\n",
        encoding="utf-8",
    )
    metrics = metrics if isinstance(metrics, dict) else {}
    metrics["model"] = {
        "llm_calls": usage["text"]["call_count"],
        "vlm_calls": usage["vision"]["call_count"],
        "billed_call_count": summary["exact_billed_call_count"],
        "unbilled_call_count": summary["pending_billing_call_count"],
        "failed_transport_round_count": summary["failed_transport_rounds"],
        "billing_amount_microusd": summary["billing_amount_microusd"],
        "billing_amount_usd": summary["billing_amount_usd"],
        "billing_reconciliation_status": "complete",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics) + "\n",
        encoding="utf-8",
    )
    manifest = manifest if isinstance(manifest, dict) else {}
    manifest["model_usage"] = usage
    names = {
        "billing_reconciliation_sha256": "billing_reconciliation.json",
        "billing_post_reconciliation_sha256": "billing_post_reconciliation.json",
        "api_call_costs_json_sha256": "api_call_costs.json",
        "api_call_costs_csv_sha256": "api_call_costs.csv",
        "model_usage_sha256": "model_usage.json",
        "metrics_sha256": "metrics.json",
    }
    manifest["result_artifact_hashes"] = {
        key: formal_eval_module._sha256_file(output_dir / name)
        for key, name in names.items()
    }
    return metrics, manifest, usage


def _complete_billing_usage():
    def call(scope, request_id, amount, *, started_at):
        return {
            "kind": f"remote_{scope}_generation",
            "role": "coder" if scope == "text" else "vlm_failure_analyzer",
            "artifact_id": f"{scope}-artifact",
            "response_index": 1,
            "request_id": request_id,
            "requested_model": "model",
            "resolved_model": "model",
            "response_model": "model",
            "canonical_resolved_model": "model",
            "started_at_unix": started_at,
            "http_status": 200,
            "input_tokens": 10,
            "output_tokens": 2,
            "cached_input_tokens": 0,
            "image_input_tokens": 0,
            "reasoning_tokens": 0,
            "billing_amount_microusd": amount,
            "billing_amount_usd": amount / 1_000_000.0,
            "billing_currency": "USD",
        }

    return {
        "text": {"calls": [call("text", "req-text", 123, started_at=1.0)]},
        "vision": {"calls": [call("vision", "req-vision", 456, started_at=2.0)]},
        "roles": {"model_name": "model", "roles": {}},
    }


def test_formal_matrix_is_three_models_by_five_unique_runs():
    paths = {
        formal_run_output_dir(
            REPO_ROOT,
            model_or_key=model.key,
            run_index=run_index,
        )
        for model in FORMAL_BACKBONES
        for run_index in range(FORMAL_RUN_COUNT)
    }

    assert len(FORMAL_BACKBONES) == 3
    assert len(paths) == 15
    assert all(
        f"/formal/{FORMAL_EVAL_PROTOCOL_ID}/backbone/" in str(path)
        for path in paths
    )


def test_nonformal_runner_defaults_memory_to_output_directory(monkeypatch, tmp_path):
    output_dir = tmp_path / "engineering"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_cap_box_carry_evolution.py", "--output-dir", str(output_dir)],
    )

    args = parse_args()

    expected = formal_memory_paths(output_dir)
    assert args.raw_attempt_memory == expected["raw_attempt_memory"]
    assert args.strategy_memory == expected["strategy_memory"]
    assert args.action_exemplar_memory == expected["action_exemplar_memory"]
    assert args.episode_manifest == expected["episode_manifest"]


def test_formal_runner_applies_frozen_budget_and_isolated_memory(monkeypatch):
    model = FORMAL_BACKBONES[0]
    output_dir = formal_run_output_dir(
        REPO_ROOT,
        model_or_key=model.key,
        run_index=0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cap_box_carry_evolution.py",
            "--eval-protocol",
            FORMAL_EVAL_PROTOCOL_ID,
            "--run-index",
            "0",
            "--seed",
            "0",
            "--planner-backend",
            "openai_compatible",
            "--planner-model",
            model.requested_model,
            "--output-dir",
            str(output_dir),
            "--execute",
        ],
    )

    args = parse_args()

    assert args.max_attempts == FORMAL_MODEL_ATTEMPT_BUDGET
    assert args.max_new_tokens == FORMAL_MAX_NEW_TOKENS
    assert args.iteration_budget == FORMAL_PHYSICAL_BUDGET
    assert args.require_robots_exit
    assert args.route_discovery
    assert args.planner_base_url == ""
    assert args.planner_network_retries == FORMAL_PROVIDER_NETWORK_RETRIES
    assert args.keep_term4_on_transient_provider_error is True
    assert args.provider_recovery_backoff_s == pytest.approx(60.0)
    assert args.provider_recovery_wall_time_cap_s == pytest.approx(1800.0)
    assert args.transport_distance_m == 1.0
    assert args.transport_speed_mps == 0.50
    assert args.team_sync_longitudinal_kp == 2.0
    assert args.post_action_stable_s == 1.0
    assert args.record_action_videos is False
    assert args.action_video_fps == 10.0
    assert args.runtime_state_file == output_dir / "runtime_state.json"
    assert args.strategy_memory == output_dir / "memory" / "strategy_specs.json"


def test_formal_runner_rejects_resume(monkeypatch):
    model = FORMAL_BACKBONES[0]
    output_dir = formal_run_output_dir(
        REPO_ROOT,
        model_or_key=model.key,
        run_index=0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cap_box_carry_evolution.py",
            "--eval-protocol",
            FORMAL_EVAL_PROTOCOL_ID,
            "--run-index",
            "0",
            "--planner-backend",
            "openai_compatible",
            "--planner-model",
            model.requested_model,
            "--output-dir",
            str(output_dir),
            "--execute",
            "--resume-checkpoint-id",
            "accepted_step_03",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_diagnostic_formal_contracts_require_non_paper_resume(monkeypatch, tmp_path):
    output_dir = REPO_ROOT / "results" / "deployment" / "non_paper" / "resume"
    resume_attempt = tmp_path / "attempt_56"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cap_box_carry_evolution.py",
            "--planner-backend",
            "openai_compatible",
            "--planner-model",
            FORMAL_BACKBONES[0].requested_model,
            "--output-dir",
            str(output_dir),
            "--execute",
            "--resume-checkpoint-id",
            "accepted_step_26",
            "--resume-attempt-dir",
            str(resume_attempt),
            "--diagnostic-formal-host-contracts",
        ],
    )

    args = parse_args()

    assert args.eval_protocol == ""
    assert args.diagnostic_formal_host_contracts is True
    assert args.output_dir == output_dir


def test_diagnostic_formal_contracts_reject_formal_output_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cap_box_carry_evolution.py",
            "--output-dir",
            str(REPO_ROOT / "results" / "deployment" / "formal" / "resume"),
            "--execute",
            "--resume-checkpoint-id",
            "accepted_step_26",
            "--resume-attempt-dir",
            str(tmp_path / "attempt_56"),
            "--diagnostic-formal-host-contracts",
        ],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_formal_runner_overrides_behavior_changing_cli_knobs(monkeypatch):
    model = FORMAL_BACKBONES[0]
    output_dir = formal_run_output_dir(
        REPO_ROOT,
        model_or_key=model.key,
        run_index=1,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cap_box_carry_evolution.py",
            "--eval-protocol",
            FORMAL_EVAL_PROTOCOL_ID,
            "--run-index",
            "1",
            "--planner-backend",
            "openai_compatible",
            "--planner-model",
            model.requested_model,
            "--output-dir",
            str(output_dir),
            "--execute",
            "--planner-base-url",
            "https://example.invalid/v1",
            "--planner-network-retries",
            "0",
            "--transport-distance-m",
            "0.1",
            "--transport-speed-mps",
            "0.1",
            "--team-sync-longitudinal-kp",
            "9.0",
            "--post-action-stable-s",
            "9.0",
            "--visual-frame",
            "/tmp/stale.png",
            "--no-record-action-videos",
            "--action-video-fps",
            "2",
        ],
    )

    args = parse_args()

    assert args.seed == 1
    assert args.planner_base_url == "https://example.invalid/v1"
    assert args.planner_network_retries == FORMAL_PROVIDER_NETWORK_RETRIES
    assert args.transport_distance_m == 1.0
    assert args.transport_speed_mps == 0.50
    assert args.team_sync_longitudinal_kp == 2.0
    assert args.post_action_stable_s == 1.0
    assert args.visual_frame is None
    assert args.record_action_videos is False
    assert args.action_video_fps == 10.0


def test_formal_freshness_rejects_existing_memory(tmp_path):
    output_dir = tmp_path / "seed_000"
    paths = formal_memory_paths(output_dir)
    paths["strategy_memory"].parent.mkdir(parents=True)
    paths["strategy_memory"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(FormalEvalConfigurationError, match="empty dynamic/action"):
        validate_fresh_formal_run(
            SimpleNamespace(
                eval_protocol=FORMAL_EVAL_PROTOCOL_ID,
                output_dir=output_dir,
            )
        )


def test_formal_freeze_commit_must_match_protocol_tag(monkeypatch, tmp_path):
    args = SimpleNamespace(eval_protocol=FORMAL_EVAL_PROTOCOL_ID)
    monkeypatch.setattr(
        formal_eval_module,
        "_git_revision",
        lambda _root, _revision: "abc123",
    )

    validate_formal_freeze_commit(
        args,
        repo_root=tmp_path,
        codebase_snapshot={"git_head": "abc123"},
    )
    with pytest.raises(FormalEvalConfigurationError, match="must equal"):
        validate_formal_freeze_commit(
            args,
            repo_root=tmp_path,
            codebase_snapshot={"git_head": "different"},
        )


def test_formal_initial_runtime_requires_frozen_payload_and_arm_setup():
    args = SimpleNamespace(eval_protocol=FORMAL_EVAL_PROTOCOL_ID)
    state = {
        "payload": {"mass_kg": 0.5, "rearward_offset_m": 0.10},
        "upper_body_pose_lock": {
            "pose_name": "elbow60_hand60_carry_ready",
            "target_pose_name": "elbow60_hand60_carry_ready",
            "kp": 24.0,
            "kd": 1.5,
            "configured_ramp_s": 6.0,
            "palm_tilt_deg": 60.0,
        },
        "mujoco_solver": "newton",
        "target": {
            "center_x_m": -1.5,
            "center_y_m": 7.6,
            "center_z_m": 0.3,
            "radius_m": 0.3,
            "half_height_m": 0.3,
        },
    }

    validate_formal_initial_runtime(args, state)
    state["payload"]["mass_kg"] = 1.0
    with pytest.raises(FormalEvalConfigurationError, match="payload.mass_kg"):
        validate_formal_initial_runtime(args, state)


def test_formal_initial_runtime_rejects_different_mujoco_solver():
    args = SimpleNamespace(eval_protocol=FORMAL_EVAL_PROTOCOL_ID)
    state = {
        "payload": {"mass_kg": 0.5, "rearward_offset_m": 0.10},
        "upper_body_pose_lock": {
            "pose_name": "elbow60_hand60_carry_ready",
            "target_pose_name": "elbow60_hand60_carry_ready",
            "kp": 24.0,
            "kd": 1.5,
            "configured_ramp_s": 6.0,
            "palm_tilt_deg": 60.0,
        },
        "mujoco_solver": "cg",
        "target": {
            "center_x_m": -1.5,
            "center_y_m": 7.6,
            "center_z_m": 0.3,
            "radius_m": 0.3,
            "half_height_m": 0.3,
        },
    }

    with pytest.raises(FormalEvalConfigurationError, match="mujoco_solver"):
        validate_formal_initial_runtime(args, state)


def test_formal_plan_preparation_does_not_modify_nonfresh_directory(tmp_path):
    output_dir = tmp_path / "seed_000"
    output_dir.mkdir()
    attempt = output_dir / "attempt_01"
    attempt.mkdir()
    sentinel = output_dir / "run_plan.json"
    sentinel.write_text("old-plan\n", encoding="utf-8")

    status, _ = prepare_run_directory(
        output_dir, {"protocol_id": FORMAL_EVAL_PROTOCOL_ID}
    )

    assert status == "not_fresh"
    assert sentinel.read_text(encoding="utf-8") == "old-plan\n"


def test_terminal_target_metrics_use_physical_support_footprint():
    task = BoxCarryTask(
        task_id="formal",
        target="target",
        target_xy_tolerance_m=0.18,
        target_z_tolerance_m=0.08,
        max_terminal_payload_tilt_deg=20.0,
    )
    target_z = (
        task.target_center_z_m
        + task.target_scale_z_m / 2.0
        + task.payload_half_height_m
    )
    contacts = {
        "target": True,
        "floor": False,
        "walls": [],
        "robots": [],
        "support_robots": [],
    }
    robots = {
        "front": {"x": 0.0, "y": 0.0},
        "back": {"x": 0.0, "y": 1.0},
    }

    upright = _target_metrics(
        task,
        {
            "x": task.target_center_x_m,
            "y": task.target_center_y_m,
            "z": target_z,
            "tilt_deg": 2.0,
            "released": True,
            "contacts": contacts,
        },
        robots,
    )
    overturned = _target_metrics(
        task,
        {
            "x": task.target_center_x_m,
            "y": task.target_center_y_m,
            "z": target_z,
            "tilt_deg": 179.96,
            "released": True,
            "contacts": contacts,
        },
        robots,
    )
    outside_quality_tolerance = _target_metrics(
        task,
        {
            "x": task.target_center_x_m + 0.19,
            "y": task.target_center_y_m,
            "z": target_z,
            "tilt_deg": 2.0,
            "released": True,
            "contacts": contacts,
        },
        robots,
    )
    outside_support_footprint = _target_metrics(
        task,
        {
            "x": task.target_center_x_m + 0.31,
            "y": task.target_center_y_m,
            "z": target_z,
            "tilt_deg": 2.0,
            "released": True,
            "contacts": contacts,
        },
        robots,
    )

    assert upright["payload_on_target"] == 1.0
    assert overturned["payload_on_target"] == 0.0
    assert outside_quality_tolerance["payload_target_xy_within_tolerance"] == 0.0
    assert outside_quality_tolerance["payload_target_com_over_support"] == 1.0
    assert outside_quality_tolerance["payload_on_target"] == 1.0
    assert outside_support_footprint["payload_target_com_over_support"] == 0.0
    assert outside_support_footprint["payload_on_target"] == 0.0


def test_phase_completion_does_not_infer_skipped_topology_from_final_pose():
    task = BoxCarryTask(
        task_id="formal",
        target="target",
        task_stage="full_scene",
    )

    metrics = _phase_completion_metrics(
        task,
        {
            "payload": {
                "active": True,
                "x": task.stage1_goal_x_m,
                "y": 3.0,
                "contacts": {},
            }
        },
        task_complete=False,
    )

    assert metrics["completed"] == 0
    assert metrics["total"] == 6
    assert metrics["fraction"] == 0.0
    assert metrics["source"] == "trusted_checkpoint_physics"


def test_phase_completion_total_follows_scene_route_configuration():
    task = BoxCarryTask(
        task_id="two_gate_scene",
        target="target",
        task_stage="full_scene",
        route_progress_gates=(
            RouteProgressGate("entry", 0.0, 1.0, 1.0),
            RouteProgressGate("destination_lane", 1.0, 0.0, 3.0),
        ),
    )

    metrics = _phase_completion_metrics(
        task,
        {"payload": {"active": True, "x": 0.0, "y": 0.0, "contacts": {}}},
        task_complete=False,
    )

    assert metrics["total"] == 4


def test_formal_summary_uses_capped_failures_and_wilson_interval():
    rows = [
        {
            "requested_model": "model",
            "metrics_available": True,
            "protocol_valid": True,
            "cost_complete": True,
            "task_success": True,
            "cpt_at_80": 12,
            "phase_completion": 1.0,
            "rejected_transactions": 3,
            "fm_calls": 10,
            "input_tokens": 100,
            "output_tokens": 20,
            "model_latency_s": 4.0,
            "api_cost_usd": 0.4,
            "wall_time_s": 40.0,
        },
        {
            "requested_model": "model",
            "metrics_available": True,
            "protocol_valid": True,
            "cost_complete": True,
            "task_success": False,
            "cpt_at_80": 80,
            "phase_completion": 0.5,
            "rejected_transactions": 8,
            "fm_calls": 20,
            "input_tokens": 200,
            "output_tokens": 40,
            "model_latency_s": 8.0,
            "api_cost_usd": 0.6,
            "wall_time_s": 80.0,
        },
    ]

    summary = summarize_rows(rows)
    lower, upper = _wilson_interval(1, 2)

    assert summary["success_rate"] == 0.5
    assert summary["cpt_at_80"] == 46.0
    assert summary["phase_completion"] == 0.75
    assert summary["cost_complete_runs"] == 2
    assert summary["api_cost_usd_total"] == pytest.approx(1.0)
    assert summary["success_rate_wilson_95_low"] == lower
    assert summary["success_rate_wilson_95_high"] == upper


def test_formal_summary_excludes_protocol_invalid_rows():
    valid = {
        "requested_model": "model",
        "metrics_available": True,
        "protocol_valid": True,
        "cost_complete": True,
        "task_success": True,
        "cpt_at_80": 12,
        "phase_completion": 1.0,
        "rejected_transactions": 3,
        "fm_calls": 10,
        "input_tokens": 100,
        "output_tokens": 20,
        "model_latency_s": 4.0,
        "api_cost_usd": 0.4,
        "wall_time_s": 40.0,
    }
    invalid = {
        **valid,
        "protocol_valid": False,
        "cost_complete": False,
        "task_success": False,
        "cpt_at_80": 80,
        "api_cost_usd": 99.0,
    }

    summary = summarize_rows([valid, invalid])

    assert summary["available_runs"] == 2
    assert summary["protocol_valid_runs"] == 1
    assert summary["cost_complete_runs"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["cpt_at_80"] == 12.0
    assert summary["api_cost_usd_total"] == pytest.approx(0.4)


def test_formal_summary_withholds_total_when_valid_cost_is_incomplete():
    complete = {
        "requested_model": "model",
        "metrics_available": True,
        "protocol_valid": True,
        "cost_complete": True,
        "task_success": True,
        "cpt_at_80": 12,
        "phase_completion": 1.0,
        "rejected_transactions": 3,
        "fm_calls": 10,
        "input_tokens": 100,
        "output_tokens": 20,
        "model_latency_s": 4.0,
        "api_cost_usd": 0.4,
        "wall_time_s": 40.0,
    }
    incomplete = {
        **complete,
        "cost_complete": False,
        "api_cost_usd": 0.6,
    }

    summary = summarize_rows([complete, incomplete])

    assert summary["protocol_valid_runs"] == 2
    assert summary["cost_complete_runs"] == 1
    assert summary["api_cost_usd_total"] is None


def _write_formal_reconciliation_fixture(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = []
    records = []
    steps = []
    memory_events = []
    outcomes = ("ACCEPT_SETUP", "TASK_COMPLETE")
    for attempt, outcome in enumerate(outcomes, start=1):
        rollout_id = f"rollout-{attempt}"
        before = f"accepted_step_{attempt - 1:02d}"
        after = f"accepted_step_{attempt:02d}"
        event = {
            "kind": "physical_rollout_finalized",
            "rollout_id": rollout_id,
            "attempt": attempt,
            "script_id": f"policy-{attempt}",
            "physically_dispatched": True,
            "physical_transaction_completed": True,
            "verifier": {
                "step_accepted": True,
                "checkpoint_before": before,
                "checkpoint_after": after,
                "transition_outcome": outcome,
            },
        }
        raw.append(event)
        memory_events.extend(
            (
                dict(event),
                {
                    "kind": "memory_update_result",
                    "rollout_id": rollout_id,
                    "attempt": attempt,
                    "status": "not_promoted",
                },
            )
        )
        records.append(
            {
                "policy": {
                    "attempt": attempt,
                    "script_id": f"policy-{attempt}",
                },
                "evaluation": {
                    "evidence": {
                        "checkpoint_outcome": {"physically_dispatched": True}
                    }
                },
                "checkpoint_before": before,
                "checkpoint_after": after,
            }
        )
        steps.append(
            {
                "index": attempt,
                "attempt": attempt,
                "script_id": f"policy-{attempt}",
                "checkpoint_before": before,
                "checkpoint_after": after,
                "transition_outcome": outcome,
                "robot_scripts": {"front": {}, "back": {}},
            }
        )
        attempt_dir = output_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir()
        (attempt_dir / "physical_transaction_commit.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "attempt": attempt,
                    "script_id": f"policy-{attempt}",
                    "checkpoint_before": before,
                    "checkpoint_after": after,
                    "checkpoint_committed": True,
                    "outcome_finalized": True,
                    "memory_required": True,
                    "status": "sealed",
                    "stages": {
                        "dispatch_confirmed": True,
                        "outcome": True,
                        "controller_accounting": True,
                        "checkpoint_lineage": True,
                        "attempt_artifacts": True,
                        "raw_rollout": True,
                        "evolution_record": True,
                        "memory_update": True,
                        "terminal_metrics": True,
                    },
                    "stage_details": {
                        "raw_rollout": {"rollout_id": rollout_id},
                        "memory_update": {"rollout_id": rollout_id},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (output_dir / "raw_rollout_evidence.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in raw),
        encoding="utf-8",
    )
    (output_dir / "evolution_records.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    memory_dir = output_dir / "memory"
    memory_dir.mkdir()
    (memory_dir / "raw_rollouts.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in memory_events),
        encoding="utf-8",
    )
    (memory_dir / "accepted_episode_manifest.json").write_text(
        json.dumps(
            {
                "kind": "accepted_checkpoint_episode",
                "schema_version": 1,
                "initial_checkpoint": "accepted_step_00",
                "latest_checkpoint": "accepted_step_02",
                "initial_state": {},
                "steps": steps,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    positive_source = {
        "attempt": 2,
        "script_id": "policy-2",
        "checkpoint_before": "accepted_step_01",
        "checkpoint_after": "accepted_step_02",
    }
    (memory_dir / "action_exemplars.json").write_text(
        json.dumps(
            {
                "kind": "verified_action_exemplar_memory",
                "schema_version": 1,
                "selection_policy": "test fixture",
                "exemplars": [
                    {
                        "id": "action-test",
                        "status": "verifier_accepted_at_source_checkpoint",
                        "method": "linear",
                        "semantic_objective": "corridor",
                        "coordination_mode": "sync",
                        "robot_calls": {"front": {}, "back": {}},
                        "source": positive_source,
                        "reuse_contract": "test fixture",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (memory_dir / "strategy_specs.json").write_text(
        json.dumps(
            {
                "kind": "deployment_decision_memory",
                "schema_version": 2,
                "description": "test fixture",
                "specs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    final_summary_path = output_dir / "final_summary.json"
    final_summary_path.write_text(
        json.dumps({"status": "SUCCESS"}) + "\n",
        encoding="utf-8",
    )
    return {
        "metrics": {
            "physical_iterations": 2,
            "accepted_step_count": 2,
            "rejected_step_count": 0,
            "terminal": {"status": "SUCCESS"},
        },
        "manifest": {
            "status": "SUCCESS",
            "artifact_hashes": {
                "final_online_memory_sha256": formal_eval_module._sha256_paths(
                    formal_memory_paths(output_dir).values()
                )
            },
            "result_artifact_hashes": {
                "final_summary_sha256": formal_eval_module._sha256_file(
                    final_summary_path
                )
            },
        },
    }


def test_formal_artifact_reconciliation_accepts_complete_canonical_ledger(tmp_path):
    payload = _write_formal_reconciliation_fixture(tmp_path)

    assert _artifact_protocol_invalid_reasons(
        output_dir=tmp_path,
        **payload,
    ) == []


def test_success_requires_task_complete_on_final_canonical_rollout(tmp_path):
    payload = _write_formal_reconciliation_fixture(tmp_path)
    raw_path = tmp_path / "raw_rollout_evidence.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    raw[0]["verifier"]["transition_outcome"] = "TASK_COMPLETE"
    raw[1]["verifier"]["transition_outcome"] = "ACCEPT_PROGRESS"
    raw_path.write_text(
        "".join(json.dumps(item) + "\n" for item in raw),
        encoding="utf-8",
    )

    reasons = _artifact_protocol_invalid_reasons(
        output_dir=tmp_path,
        **payload,
    )

    assert "SUCCESS_FINAL_ROLLOUT_NOT_TASK_COMPLETE" in reasons


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            lambda journal: journal.pop("schema_version"),
            "TRANSACTION_JOURNAL_SCHEMA_MISMATCH:attempt_01",
        ),
        (
            lambda journal: journal["stages"].pop("controller_accounting"),
            "TRANSACTION_JOURNAL_STAGE_CONTRACT_MISMATCH:attempt_01",
        ),
        (
            lambda journal: journal.pop("outcome_finalized"),
            "TRANSACTION_JOURNAL_UNSEALED:attempt_01",
        ),
    ],
)
def test_formal_artifact_reconciliation_rejects_legacy_transaction_journal(
    tmp_path,
    mutation,
    expected_reason,
):
    payload = _write_formal_reconciliation_fixture(tmp_path)
    journal_path = tmp_path / "attempt_01" / "physical_transaction_commit.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    mutation(journal)
    journal_path.write_text(json.dumps(journal) + "\n", encoding="utf-8")

    reasons = _artifact_protocol_invalid_reasons(output_dir=tmp_path, **payload)

    assert expected_reason in reasons


def test_formal_artifact_reconciliation_accepts_rejected_restored_parent_identity(
    tmp_path,
):
    payload = _write_formal_reconciliation_fixture(tmp_path)
    raw_path = tmp_path / "raw_rollout_evidence.jsonl"
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    raw[0]["verifier"]["transition_outcome"] = "ACCEPT_PROGRESS"
    raw[1]["verifier"].update(
        {
            "step_accepted": False,
            "checkpoint_after": "accepted_step_01",
            "transition_outcome": "REJECT_UNSAFE",
        }
    )
    raw_path.write_text(
        "".join(json.dumps(item) + "\n" for item in raw),
        encoding="utf-8",
    )

    memory_path = tmp_path / "memory" / "raw_rollouts.jsonl"
    memory_events = [
        json.loads(line) for line in memory_path.read_text(encoding="utf-8").splitlines()
    ]
    for event in memory_events:
        if event.get("kind") != "physical_rollout_finalized":
            continue
        if event.get("attempt") == 1:
            event["verifier"]["transition_outcome"] = "ACCEPT_PROGRESS"
        elif event.get("attempt") == 2:
            event["verifier"].update(raw[1]["verifier"])
    memory_path.write_text(
        "".join(json.dumps(item) + "\n" for item in memory_events),
        encoding="utf-8",
    )

    records_path = tmp_path / "evolution_records.jsonl"
    records = [
        json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    records[1]["checkpoint_after"] = "accepted_step_01"
    records_path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )

    journal_path = tmp_path / "attempt_02" / "physical_transaction_commit.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal.update(
        {
            "checkpoint_after": "accepted_step_01",
            "checkpoint_committed": False,
            "outcome_finalized": True,
            "step_accepted": False,
        }
    )
    journal_path.write_text(json.dumps(journal) + "\n", encoding="utf-8")

    episode_path = tmp_path / "memory" / "accepted_episode_manifest.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["steps"] = episode["steps"][:1]
    episode["steps"][0]["transition_outcome"] = "ACCEPT_PROGRESS"
    episode["latest_checkpoint"] = "accepted_step_01"
    episode_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    positive_source = {
        "attempt": 1,
        "script_id": "policy-1",
        "checkpoint_before": "accepted_step_00",
        "checkpoint_after": "accepted_step_01",
    }
    (tmp_path / "memory" / "action_exemplars.json").write_text(
        json.dumps(
            {
                "kind": "verified_action_exemplar_memory",
                "schema_version": 1,
                "selection_policy": "test fixture",
                "exemplars": [
                    {
                        "id": "action-test",
                        "status": "verifier_accepted_at_source_checkpoint",
                        "method": "linear",
                        "semantic_objective": "corridor",
                        "coordination_mode": "sync",
                        "robot_calls": {"front": {}, "back": {}},
                        "source": positive_source,
                        "reuse_contract": "test fixture",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "memory" / "strategy_specs.json").write_text(
        json.dumps(
            {
                "kind": "deployment_decision_memory",
                "schema_version": 2,
                "description": "test fixture",
                "specs": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload["metrics"].update(
        {"accepted_step_count": 1, "rejected_step_count": 1}
    )
    payload["manifest"]["status"] = "BACKEND_FATAL"
    payload["manifest"]["artifact_hashes"]["final_online_memory_sha256"] = (
        formal_eval_module._sha256_paths(formal_memory_paths(tmp_path).values())
    )

    assert _artifact_protocol_invalid_reasons(
        output_dir=tmp_path,
        **payload,
    ) == []


def test_formal_artifact_reconciliation_detects_v70_attempt_44_split_commit(
    tmp_path,
):
    payload = _write_formal_reconciliation_fixture(tmp_path)
    payload["metrics"]["physical_iterations"] = 1
    payload["metrics"]["accepted_step_count"] = 1
    records = (tmp_path / "evolution_records.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    (tmp_path / "evolution_records.jsonl").write_text(
        records[0] + "\n",
        encoding="utf-8",
    )
    journal = tmp_path / "attempt_02" / "physical_transaction_commit.json"
    journal_payload = json.loads(journal.read_text(encoding="utf-8"))
    journal_payload["status"] = "pending"
    journal_payload["stages"]["terminal_metrics"] = False
    journal.write_text(json.dumps(journal_payload) + "\n", encoding="utf-8")

    reasons = _artifact_protocol_invalid_reasons(output_dir=tmp_path, **payload)

    assert "TRANSACTION_JOURNAL_UNSEALED:attempt_02" in reasons
    assert any(reason.startswith("PHYSICAL_COUNT_MISMATCH:raw=2,evolution=1") for reason in reasons)
    assert any(reason.startswith("ACCEPTED_COUNT_MISMATCH:raw=2,manifest=2,metrics=1") for reason in reasons)


def test_formal_artifact_reconciliation_rejects_setup_memory_promotion(tmp_path):
    payload = _write_formal_reconciliation_fixture(tmp_path)
    path = tmp_path / "memory" / "action_exemplars.json"
    action_memory = json.loads(path.read_text(encoding="utf-8"))
    action_memory["exemplars"][0]["source"] = {
        "attempt": 1,
        "script_id": "policy-1",
        "checkpoint_before": "accepted_step_00",
        "checkpoint_after": "accepted_step_01",
    }
    path.write_text(json.dumps(action_memory) + "\n", encoding="utf-8")

    reasons = _artifact_protocol_invalid_reasons(output_dir=tmp_path, **payload)

    assert "ACTION_EXEMPLAR_NOT_FROM_POSITIVE_CANONICAL_ROLLOUT" in reasons


def test_system_stall_reconciliation_accepts_three_model_stall_cycles(tmp_path):
    events = [
        {
            "checkpoint_id": "accepted_step_11",
            "generation_cycle": cycle,
            "counts_as_model_stall": True,
            "physical_dispatch": False,
            "checkpoint_change": False,
            "system_stall": cycle == 3,
        }
        for cycle in (1, 2, 3)
    ]
    (tmp_path / "generation_stalls.json").write_text(
        json.dumps({"events": events}) + "\n",
        encoding="utf-8",
    )

    assert _system_stall_evidence_invalid_reasons(
        tmp_path,
        terminal={"error": "three consecutive complete generation cycles stalled"},
    ) == []


def test_system_stall_reconciliation_accepts_repeated_local_empty_scope(tmp_path):
    events = [
        {
            "checkpoint_id": "accepted_step_11",
            "generation_cycle": cycle,
            "cycle_outcome": "MUTATION_SCOPE_INFEASIBLE",
            "selected_mutation_level": "parameter_tuning",
            "counts_as_model_stall": False,
            "physical_dispatch": False,
            "checkpoint_change": False,
            "system_stall": cycle == 3,
        }
        for cycle in (1, 2, 3)
    ]
    (tmp_path / "generation_stalls.json").write_text(
        json.dumps({"events": events}) + "\n",
        encoding="utf-8",
    )

    assert _system_stall_evidence_invalid_reasons(
        tmp_path,
        terminal={
            "error": (
                "three complete generation cycles repeatedly selected local "
                "scopes while another recovery level remained unproven"
            )
        },
    ) == []


def test_system_stall_reconciliation_rejects_three_distinct_empty_scopes(tmp_path):
    events = [
        {
            "checkpoint_id": "accepted_step_11",
            "generation_cycle": cycle,
            "cycle_outcome": "MUTATION_SCOPE_INFEASIBLE",
            "selected_mutation_level": level,
            "counts_as_model_stall": False,
            "physical_dispatch": False,
            "checkpoint_change": False,
            "system_stall": False,
        }
        for cycle, level in enumerate(
            ("parameter_tuning", "coordination_mutation", "strategy_mutation"),
            start=1,
        )
    ]
    (tmp_path / "generation_stalls.json").write_text(
        json.dumps({"events": events}) + "\n",
        encoding="utf-8",
    )

    reasons = _system_stall_evidence_invalid_reasons(
        tmp_path,
        terminal={"error": "another recovery level remained unproven"},
    )

    assert reasons == ["SYSTEM_STALL_EVIDENCE_INCOMPLETE_OR_INCOHERENT"]


def test_formal_manifest_records_protocol_hashes_and_no_secret_field(tmp_path):
    model = FORMAL_BACKBONES[0]
    output_dir = tmp_path / "seed_000"
    memory = formal_memory_paths(output_dir)
    for path in memory.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    route_spec = tmp_path / "route.json"
    route_spec.write_text('{"kind":"directed_connected_route"}\n', encoding="utf-8")
    args = SimpleNamespace(
        eval_protocol=FORMAL_EVAL_PROTOCOL_ID,
        output_dir=output_dir,
        planner_model=model.requested_model,
        planner_backend=model.planner_backend,
        planner_api_key_env="COEVOLVE_API_KEY",
        run_index=0,
        seed=0,
        physical_safety_profile="task",
        route_spec=route_spec,
    )
    task = BoxCarryTask(task_id="formal", target="target", require_robots_exit=True)
    snapshot = {
        "git_head": "abc123",
        "git_branch": "main",
        "git_dirty_entries": [],
        "source_sha256": "source-hash",
        "files": {
            "src/praxis_coop/cap_policy/prompts.py": "prompt-hash",
            "src/praxis_coop/cap_policy/route_spec.py": "route-source-hash",
            "src/praxis_coop/cap_policy/transition_verifier.py": (
                "transition-source-hash"
            ),
        },
    }

    path = write_formal_run_manifest(
        args=args,
        repo_root=REPO_ROOT,
        codebase_snapshot=snapshot,
        task=task,
        status="initialized",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    initial_memory_hash = payload["artifact_hashes"]["initial_online_memory_sha256"]

    memory["strategy_memory"].write_text('{"learned": true}\n', encoding="utf-8")
    (output_dir / "billing_reconciliation.json").write_text(
        '{"status": "complete", "remaining": 0}\n',
        encoding="utf-8",
    )
    path = write_formal_run_manifest(
        args=args,
        repo_root=REPO_ROOT,
        codebase_snapshot=snapshot,
        task=task,
        status="SUCCESS",
        model_usage={
            "text": {
                "calls": [{"resolved_model": model.requested_model}],
            }
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["protocol"]["run_count"] == 5
    assert payload["budgets"]["physical_transactions"] == 80
    assert payload["artifact_hashes"]["prompt_sha256"] == "prompt-hash"
    assert payload["route_spec_sha256"] == payload["artifact_hashes"][
        "route_spec_sha256"
    ]
    assert len(payload["route_spec_sha256"]) == 64
    assert len(payload["artifact_hashes"]["verifier_source_sha256"]) == 64
    assert (
        payload["artifact_hashes"]["initial_online_memory_sha256"]
        == initial_memory_hash
    )
    assert (
        payload["artifact_hashes"]["final_online_memory_sha256"] != initial_memory_hash
    )
    assert len(payload["evaluation_fingerprint_sha256"]) == 64
    assert "api_key" not in json.dumps(payload).lower()
    assert len(payload["manifest_sha256"]) == 64
    metrics = {
        "task_success": True,
        "physical_iterations": 12,
        "model": {
            "llm_calls": 1,
            "vlm_calls": 1,
            "unbilled_call_count": 7,
            "billing_reconciliation_status": "incomplete",
        },
        "terminal": {"status": "SUCCESS", "codebase_unchanged": True},
    }
    assert _is_protocol_valid_run(
        metrics=metrics,
        manifest=payload,
        expected_run_index=0,
        expected_seed=0,
        expected_model=model.requested_model,
    )
    assert not _is_cost_complete(metrics=metrics, manifest=payload)


def test_exact_billing_reconciliation_accepts_consistent_per_call_evidence(tmp_path):
    metrics, manifest, _ = _seal_complete_billing_fixture(
        tmp_path,
        _complete_billing_usage(),
    )

    assert _billing_artifact_invalid_reasons(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    ) == []
    assert _is_cost_complete(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )


def test_exact_billing_reconciliation_rejects_duplicate_request_ids(tmp_path):
    usage = _complete_billing_usage()
    usage["vision"]["calls"][0]["request_id"] = "req-text"
    metrics, manifest, _ = _seal_complete_billing_fixture(tmp_path, usage)

    reasons = _billing_artifact_invalid_reasons(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )

    assert "MODEL_USAGE_DUPLICATE_REQUEST_ID" in reasons
    assert "API_COST_LEDGER_DUPLICATE_REQUEST_ID" in reasons
    assert not _is_cost_complete(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )


def test_exact_billing_reconciliation_rejects_completed_response_without_id(
    tmp_path,
):
    usage = _complete_billing_usage()
    usage["text"]["calls"][0]["request_id"] = ""
    metrics, manifest, _ = _seal_complete_billing_fixture(tmp_path, usage)

    reasons = _billing_artifact_invalid_reasons(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )

    assert "COMPLETED_MODEL_RESPONSE_MISSING_REQUEST_ID" in reasons
    assert not _is_cost_complete(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )


def test_exact_billing_reconciliation_rejects_ledger_call_omission(tmp_path):
    metrics, manifest, _ = _seal_complete_billing_fixture(
        tmp_path,
        _complete_billing_usage(),
    )
    ledger_path = tmp_path / "api_call_costs.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["calls"].pop()
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    manifest["result_artifact_hashes"]["api_call_costs_json_sha256"] = (
        formal_eval_module._sha256_file(ledger_path)
    )

    reasons = _billing_artifact_invalid_reasons(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )

    assert "MODEL_USAGE_LEDGER_CALL_IDENTITY_MISMATCH" in reasons
    assert "MODEL_USAGE_LEDGER_REQUEST_ID_MISMATCH" in reasons
    assert not _is_cost_complete(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )


def test_exact_billing_reconciliation_rejects_metrics_total_drift(tmp_path):
    metrics, manifest, _ = _seal_complete_billing_fixture(
        tmp_path,
        _complete_billing_usage(),
    )
    metrics["model"]["billing_amount_microusd"] += 1
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    manifest["result_artifact_hashes"]["metrics_sha256"] = (
        formal_eval_module._sha256_file(metrics_path)
    )

    reasons = _billing_artifact_invalid_reasons(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )

    assert "METRICS_MODEL_BILLING_AMOUNT_MICROUSD_MISMATCH" in reasons
    assert not _is_cost_complete(
        metrics=metrics,
        manifest=manifest,
        output_dir=tmp_path,
    )


def test_v18_system_stall_is_a_protocol_valid_failed_replicate(tmp_path):
    model = FORMAL_BACKBONES[0]
    output_dir = tmp_path / "seed_000"
    args = SimpleNamespace(
        eval_protocol=FORMAL_EVAL_PROTOCOL_ID,
        output_dir=output_dir,
        planner_model=model.requested_model,
        planner_backend=model.planner_backend,
        planner_api_key_env="COEVOLVE_API_KEY",
        run_index=0,
        seed=0,
        physical_safety_profile="task",
    )
    task = BoxCarryTask(task_id="formal", target="target", require_robots_exit=True)
    snapshot = {
        "git_head": "abc123",
        "git_branch": "main",
        "git_dirty_entries": [],
        "source_sha256": "source-hash",
        "files": {},
    }
    (output_dir / "billing_reconciliation.json").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (output_dir / "billing_reconciliation.json").write_text(
        '{"status": "complete", "remaining": 0}\n',
        encoding="utf-8",
    )
    manifest_path = write_formal_run_manifest(
        args=args,
        repo_root=REPO_ROOT,
        codebase_snapshot=snapshot,
        task=task,
        status="SYSTEM_STALL",
        model_usage={
            "text": {"calls": [{"resolved_model": model.requested_model}]}
        },
    )

    assert _is_protocol_valid_run(
        metrics={
            "task_success": False,
            "physical_iterations": 0,
            "model": {
                "llm_calls": 1,
                "vlm_calls": 0,
                "unbilled_call_count": 0,
                "billing_reconciliation_status": "complete",
            },
            "terminal": {
                "status": "SYSTEM_STALL",
                "codebase_unchanged": True,
            },
        },
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        expected_run_index=0,
        expected_seed=0,
        expected_model=model.requested_model,
    )


def test_resumed_generation_recovers_on_second_complete_cycle(tmp_path):
    tracker = GenerationStallTracker(tmp_path)
    physical_iterations = 7
    calls: list[int] = []
    retries: list[int] = []

    def generate(cycle: int) -> str:
        calls.append(cycle)
        if cycle == 1:
            raise ModelRoleStallError(
                "prose-only mutation",
                role="mutation_operator",
                artifact_id="attempt_140",
                response_count=5,
                validation_errors=("NO_EFFECTIVE_MUTATION",),
            )
        return "valid-child"

    result = _run_checkpoint_generation_cycles(
        tracker=tracker,
        checkpoint_id="accepted_step_49",
        generate=generate,
        on_retry=lambda cycle, _event: retries.append(cycle),
    )

    assert result == "valid-child"
    assert calls == [1, 2]
    assert retries == [2]
    assert physical_iterations == 7
    assert tracker.consecutive_stalls == 0


def test_resumed_generation_seals_system_stall_after_three_cycles(tmp_path):
    tracker = GenerationStallTracker(tmp_path)

    def generate(cycle: int) -> str:
        raise ModelRoleStallError(
            f"invalid cycle {cycle}",
            role="mutation_operator",
            artifact_id=f"attempt_140_cycle_{cycle}",
            response_count=5,
            validation_errors=("NO_EFFECTIVE_MUTATION",),
        )

    with pytest.raises(SystemStallError, match="accepted_step_49"):
        _run_checkpoint_generation_cycles(
            tracker=tracker,
            checkpoint_id="accepted_step_49",
            generate=generate,
        )

    assert [item["generation_cycle"] for item in tracker.events] == [1, 2, 3]
    assert all(item["physical_dispatch"] is False for item in tracker.events)


def test_empty_scope_is_not_model_blame_but_still_consumes_total_cycle(tmp_path):
    tracker = GenerationStallTracker(tmp_path)
    calls: list[int] = []

    def generate(cycle: int) -> str:
        calls.append(cycle)
        if cycle == 1:
            raise ModelRoleStallError(
                "selected scope is contract-infeasible",
                role="evolution_manager",
                artifact_id="attempt_140",
                response_count=1,
                validation_errors=("scope infeasible",),
                cycle_outcome=MUTATION_SCOPE_INFEASIBLE,
            )
        if cycle in {2, 3}:
            raise ModelRoleStallError(
                f"invalid model cycle {cycle}",
                role="coder",
                artifact_id="attempt_140",
                response_count=5,
                validation_errors=("invalid model response",),
            )
        return "fourth-cycle-must-not-run"

    with pytest.raises(SystemStallError, match="no executable child"):
        _run_checkpoint_generation_cycles(
            tracker=tracker,
            checkpoint_id="accepted_step_49",
            generate=generate,
        )

    assert calls == [1, 2, 3]
    assert [item["counts_as_model_stall"] for item in tracker.events] == [
        False,
        True,
        True,
    ]


def test_repeated_local_empty_scope_does_not_claim_global_host_contract_error(
    tmp_path,
):
    tracker = GenerationStallTracker(tmp_path)
    calls: list[int] = []

    def generate(cycle: int) -> str:
        calls.append(cycle)
        if cycle <= 2:
            raise ModelRoleStallError(
                "parameter scope is contract-infeasible",
                role="evolution_manager",
                artifact_id=f"attempt_140_cycle_{cycle}",
                response_count=1,
                validation_errors=("scope infeasible",),
                cycle_outcome=MUTATION_SCOPE_INFEASIBLE,
                selected_mutation_level="parameter_tuning",
            )
        return "strategy-remains-available"

    assert (
        _run_checkpoint_generation_cycles(
            tracker=tracker,
            checkpoint_id="accepted_step_49",
            generate=generate,
        )
        == "strategy-remains-available"
    )
    assert calls == [1, 2, 3]
    assert not _all_recovery_levels_deterministically_infeasible(tracker.events)


def test_three_repeated_empty_local_scopes_are_model_stall_not_host_error(
    tmp_path,
):
    tracker = GenerationStallTracker(tmp_path)

    def generate(cycle: int) -> str:
        raise ModelRoleStallError(
            "parameter scope is contract-infeasible",
            role="evolution_manager",
            artifact_id=f"attempt_141_cycle_{cycle}",
            response_count=1,
            validation_errors=("scope infeasible",),
            cycle_outcome=MUTATION_SCOPE_INFEASIBLE,
            selected_mutation_level="parameter_tuning",
        )

    with pytest.raises(SystemStallError, match="no executable child"):
        _run_checkpoint_generation_cycles(
            tracker=tracker,
            checkpoint_id="accepted_step_49",
            generate=generate,
        )
    assert not _all_recovery_levels_deterministically_infeasible(tracker.events)


def test_host_contract_error_requires_distinct_proof_for_every_recovery_level(
    tmp_path,
):
    tracker = GenerationStallTracker(tmp_path)
    levels = (
        "parameter_tuning",
        "coordination_mutation",
        "strategy_mutation",
    )

    def generate(cycle: int) -> str:
        level = levels[cycle - 1]
        raise ModelRoleStallError(
            f"{level} is contract-infeasible",
            role="evolution_manager",
            artifact_id=f"attempt_140_cycle_{cycle}",
            response_count=1,
            validation_errors=("scope infeasible",),
            cycle_outcome=MUTATION_SCOPE_INFEASIBLE,
            selected_mutation_level=level,
        )

    with pytest.raises(HostContractError, match="all available recovery"):
        _run_checkpoint_generation_cycles(
            tracker=tracker,
            checkpoint_id="accepted_step_49",
            generate=generate,
        )
    assert _all_recovery_levels_deterministically_infeasible(tracker.events)


def test_ancestor_rewind_keeps_complete_authoritative_state():
    rewound = {
        "payload": {"x": 1.0, "active": True},
        "robots": {"front": {"support": True}, "back": {"support": True}},
        "route_phase": {"phase_id": "route_phase_2"},
        "payload_lifecycle": {"state": "active_carry"},
        "checkpoint": {"reference_pose": {"payload": {"x": 1.0}}},
    }

    reference = _checkpoint_reference_after_ancestor_rewind(rewound)

    assert reference is rewound
    assert reference["payload_lifecycle"] == {"state": "active_carry"}
    assert reference["route_phase"] == {"phase_id": "route_phase_2"}
    assert reference != rewound["checkpoint"]["reference_pose"]


def test_physical_transaction_journal_is_atomic_and_recoverable(tmp_path):
    attempt_dir = tmp_path / "attempt_44"
    journal = _begin_physical_transaction_commit(
        attempt_dir=attempt_dir,
        attempt=44,
        script_id="policy-44",
        checkpoint_before="accepted_step_29",
        checkpoint_after="accepted_step_30",
        checkpoint_committed=True,
        memory_required=True,
    )
    assert json.loads(journal.read_text(encoding="utf-8"))["stages"] == {
        "dispatch_confirmed": True,
        "outcome": True,
        "controller_accounting": False,
        "checkpoint_lineage": False,
        "attempt_artifacts": False,
        "raw_rollout": False,
        "evolution_record": False,
        "memory_update": False,
        "terminal_metrics": False,
    }
    _advance_physical_transaction_commit(journal, "controller_accounting")
    _advance_physical_transaction_commit(journal, "checkpoint_lineage")
    _advance_physical_transaction_commit(journal, "attempt_artifacts")
    _advance_physical_transaction_commit(
        journal, "raw_rollout", details={"rollout_id": "rollout-44"}
    )
    _advance_physical_transaction_commit(
        journal,
        "evolution_record",
        details={"state": "physical_transaction_finalized_pending_generation"},
    )

    interrupted = json.loads(journal.read_text(encoding="utf-8"))
    assert interrupted["status"] == "pending"
    assert interrupted["stages"]["raw_rollout"] is True
    assert interrupted["stages"]["evolution_record"] is True
    assert interrupted["stages"]["memory_update"] is False

    _advance_physical_transaction_commit(
        journal, "memory_update", details={"status": "not_promoted"}
    )
    committed = json.loads(journal.read_text(encoding="utf-8"))
    assert committed["status"] == "artifacts_committed"

    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    _mark_transaction_journals_terminal_metrics(tmp_path)
    sealed = json.loads(journal.read_text(encoding="utf-8"))
    assert sealed["stages"]["terminal_metrics"] is True
    assert sealed["status"] == "sealed"


def test_confirmed_dispatch_journal_counts_keyboard_interrupt_before_evaluation(
    tmp_path,
):
    journal = _begin_physical_transaction_commit(
        attempt_dir=tmp_path / "attempt_07",
        attempt=7,
        script_id="policy-7",
        checkpoint_before="accepted_step_03",
        checkpoint_after="",
        checkpoint_committed=False,
        memory_required=True,
        outcome_finalized=False,
    )

    interrupted = json.loads(journal.read_text(encoding="utf-8"))
    assert interrupted["outcome_finalized"] is False
    assert interrupted["stages"]["dispatch_confirmed"] is True
    assert interrupted["stages"]["outcome"] is False
    reconciled = _controller_metrics_with_dispatch_journals(
        {
            "task_success": False,
            "physical_iterations": 0,
            "budget": {
                "maximum_physical_iterations": 80,
                "unlimited": False,
                "used": 0,
                "remaining": 80,
                "exhausted": False,
            },
        },
        tmp_path,
    )

    assert _terminal_reason_for_exception(KeyboardInterrupt()) == "OPERATOR_CANCELLED"
    assert reconciled["physical_iterations"] == 1
    assert reconciled["pending_physical_evaluation_iterations"] == 1
    assert reconciled["budget"]["used"] == 1
    assert reconciled["budget"]["remaining"] == 79


def test_rejected_transaction_binds_journal_to_restored_parent_checkpoint(tmp_path):
    journal = _begin_physical_transaction_commit(
        attempt_dir=tmp_path / "attempt_08",
        attempt=8,
        script_id="policy-8",
        checkpoint_before="accepted_step_04",
        checkpoint_after="",
        checkpoint_committed=False,
        memory_required=True,
        outcome_finalized=False,
    )

    _finalize_physical_transaction_outcome(
        journal,
        checkpoint_after="accepted_step_04",
        checkpoint_committed=False,
        step_accepted=False,
    )

    finalized = json.loads(journal.read_text(encoding="utf-8"))
    assert finalized["checkpoint_before"] == "accepted_step_04"
    assert finalized["checkpoint_after"] == "accepted_step_04"
    assert finalized["checkpoint_committed"] is False
    assert finalized["outcome_finalized"] is True
    assert finalized["stages"]["outcome"] is True
    assert finalized["stages"]["checkpoint_lineage"] is True


@pytest.mark.parametrize("interruption_stage", ["memory_update", "next_step_evidence"])
def test_physical_accounting_is_committed_before_post_evaluation_interruptions(
    tmp_path,
    interruption_stage,
):
    controller = EvolutionController(
        max_physical_iterations=80,
        output_dir=tmp_path / "controller",
        resume_history=False,
    )
    policy = PolicyScript(
        script_id="policy-9",
        attempt=9,
        code=(
            "def policy(api):\n"
            "    api.coordinated_linear_transport(direction='forward', "
            "distance_m=0.25, speed_mps=0.5, ramp_s=1.5)\n"
            "    api.stop_all()\n"
        ),
    )
    report = EvaluationReport(success=False, failure_type="no_payload_progress")
    trace = ExecutionTrace(script_id=policy.script_id, attempt=policy.attempt, status="success")

    first = controller.record_and_decide(
        policy=policy,
        report=report,
        trace=trace,
        checkpoint_id="accepted_step_05",
        physically_dispatched=True,
    )
    with pytest.raises(KeyboardInterrupt, match=interruption_stage):
        raise KeyboardInterrupt(interruption_stage)

    assert first.physical_iterations == 1
    assert controller.metrics()["physical_iterations"] == 1
    repeated = controller.record_and_decide(
        policy=policy,
        report=report,
        trace=trace,
        checkpoint_id="accepted_step_05",
        physically_dispatched=True,
    )
    assert repeated.physical_iterations == 1
    assert controller.metrics()["physical_iterations"] == 1


def test_rollout_event_append_is_idempotent_but_rejects_conflicting_duplicate(
    tmp_path,
):
    path = tmp_path / "raw_rollout_evidence.jsonl"
    event = {
        "kind": "physical_rollout_finalized",
        "rollout_id": "rollout-44",
        "attempt": 44,
        "physical_transaction_completed": True,
    }

    _append_jsonl_event(path, event)
    _append_jsonl_event(path, dict(event))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    with pytest.raises(RuntimeError, match="conflicting duplicate rollout event"):
        _append_jsonl_event(path, {**event, "attempt": 45})


def test_terminal_finalizer_writes_metrics_for_model_error(tmp_path):
    runtime_state = tmp_path / "runtime_state.json"
    runtime_state.write_text('{"payload": {}, "robots": {}}\n', encoding="utf-8")

    class Guard:
        snapshot = {}

        @staticmethod
        def verify_unchanged(*, checkpoint):
            assert checkpoint == "before_final_summary"

    class Controller:
        @staticmethod
        def metrics():
            return {
                "task_success": False,
                "physical_iterations": 0,
                "budget": {"exhausted": False},
            }

    result = _finalize_experiment(
        args=SimpleNamespace(
            output_dir=tmp_path,
            runtime_state_file=runtime_state,
            eval_protocol="",
        ),
        task=BoxCarryTask(task_id="formal_error", target="target"),
        records=[],
        experiment_initial_state={"payload": {}, "robots": {}},
        experiment_started_at=0.0,
        codebase_guard=Guard(),
        agent_client=None,
        visual_verifier=None,
        role_pipeline=None,
        evolution_controller=Controller(),
        status="SYSTEM_STALL",
        terminal_error=RuntimeError("invalid structured output"),
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert result["status"] == "SYSTEM_STALL"
    assert metrics["terminal"]["status"] == "SYSTEM_STALL"
    assert metrics["terminal"]["error_type"] == "RuntimeError"
    assert (tmp_path / "final_summary.json").is_file()
    assert (tmp_path / "model_usage.json").is_file()
    assert (tmp_path / "billing_reconciliation.json").is_file()
    assert (tmp_path / "api_call_costs.json").is_file()
    assert (tmp_path / "api_call_costs.csv").is_file()


def test_terminal_seals_outcome_before_billing_failure(tmp_path, monkeypatch):
    runtime_state = tmp_path / "runtime_state.json"
    runtime_state.write_text(
        '{"runtime_status":"running","control_health":{"healthy":true},'
        '"payload":{},"robots":{}}\n',
        encoding="utf-8",
    )

    class Guard:
        snapshot = {"git_head": "frozen"}

        @staticmethod
        def verify_unchanged(*, checkpoint):
            assert checkpoint == "before_final_summary"

    class Controller:
        @staticmethod
        def metrics():
            return {
                "task_success": False,
                "physical_iterations": 0,
                "budget": {"exhausted": False},
            }

    manifest_statuses = []

    def write_manifest(**kwargs):
        manifest_statuses.append(kwargs["status"])
        (tmp_path / "run_manifest.json").write_text(
            json.dumps({"status": kwargs["status"]}) + "\n",
            encoding="utf-8",
        )

    def fail_billing(*_args, **_kwargs):
        assert (tmp_path / "metrics.json").is_file()
        assert (tmp_path / "model_usage.json").is_file()
        assert json.loads((tmp_path / "run_manifest.json").read_text())["status"] == (
            "SYSTEM_STALL"
        )
        raise RuntimeError("billing service unavailable")

    monkeypatch.setattr(
        "scripts.run_cap_box_carry_evolution.write_formal_run_manifest",
        write_manifest,
    )
    monkeypatch.setattr(
        "scripts.run_cap_box_carry_evolution._reconcile_billing_clients",
        fail_billing,
    )
    result = _finalize_experiment(
        args=SimpleNamespace(
            output_dir=tmp_path,
            runtime_state_file=runtime_state,
            eval_protocol="",
            write_html=False,
        ),
        task=BoxCarryTask(task_id="billing_failure", target="target"),
        records=[],
        experiment_initial_state={"payload": {}, "robots": {}},
        experiment_started_at=0.0,
        codebase_guard=Guard(),
        agent_client=None,
        visual_verifier=None,
        role_pipeline=None,
        evolution_controller=Controller(),
        status="SYSTEM_STALL",
        terminal_error=RuntimeError("model generation stalled"),
    )

    billing = json.loads(
        (tmp_path / "billing_reconciliation.json").read_text(encoding="utf-8")
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert result["status"] == "SYSTEM_STALL"
    assert metrics["terminal"]["status"] == "SYSTEM_STALL"
    assert billing["status"] == "incomplete"
    assert billing["terminal_outcome_preserved"] is True
    assert manifest_statuses == ["SYSTEM_STALL", "SYSTEM_STALL"]


def test_terminal_runtime_fault_detects_control_loss():
    assert _terminal_runtime_fault({})["runtime_fault"] == {
        "type": "runtime_state_missing"
    }
    assert not _terminal_runtime_fault(
        {"runtime_status": "running", "control_health": {"healthy": True}}
    )
    fault = _terminal_runtime_fault(
        {
            "runtime_status": "control_lost",
            "control_health": {"healthy": False},
            "runtime_fault": {"type": "lowcmd_stale", "robots": ["front"]},
        }
    )

    assert fault == {
        "runtime_status": "control_lost",
        "control_healthy": False,
        "runtime_fault": {"type": "lowcmd_stale", "robots": ["front"]},
    }


def test_formal_finalizer_overrides_system_stall_when_runtime_is_lost(
    tmp_path,
    monkeypatch,
):
    runtime_state = tmp_path / "runtime_state.json"
    runtime_state.write_text(
        json.dumps(
            {
                "runtime_status": "control_lost",
                "runtime_fault": {"type": "lowcmd_stale", "robots": ["front"]},
                "control_health": {"healthy": False},
                "payload": {},
                "robots": {},
            }
        ),
        encoding="utf-8",
    )

    class Guard:
        snapshot = {}

        @staticmethod
        def verify_unchanged(*, checkpoint):
            assert checkpoint == "before_final_summary"

    class Controller:
        @staticmethod
        def metrics():
            return {
                "task_success": False,
                "physical_iterations": 0,
                "budget": {"exhausted": False},
            }

    monkeypatch.setattr(
        "scripts.run_cap_box_carry_evolution.write_formal_run_manifest",
        lambda **_kwargs: None,
    )
    result = _finalize_experiment(
        args=SimpleNamespace(
            output_dir=tmp_path,
            runtime_state_file=runtime_state,
            eval_protocol=FORMAL_EVAL_PROTOCOL_ID,
        ),
        task=BoxCarryTask(task_id="formal_backend_loss", target="target"),
        records=[],
        experiment_initial_state={"payload": {}, "robots": {}},
        experiment_started_at=0.0,
        codebase_guard=Guard(),
        agent_client=None,
        visual_verifier=None,
        role_pipeline=None,
        evolution_controller=Controller(),
        status="SYSTEM_STALL",
        terminal_error=SystemStallError("three complete cycles stalled"),
    )
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))

    assert result["status"] == "BACKEND_FATAL"
    assert metrics["terminal"]["status"] == "BACKEND_FATAL"
    assert metrics["terminal"]["error_type"] == "RuntimeControlLost"
    assert metrics["terminal"]["decision_terminal_error_type"] == (
        "SystemStallError"
    )
    assert metrics["terminal"]["runtime_healthy_at_termination"] is False
    assert metrics["terminal"]["runtime_fault_at_termination"]["runtime_fault"] == {
        "type": "lowcmd_stale",
        "robots": ["front"],
    }


def test_provider_and_role_stalls_use_distinct_terminal_reasons():
    assert (
        _terminal_reason_for_exception(ProviderFatalError("HTTP 429"))
        == "PROVIDER_FATAL"
    )
    assert (
        _terminal_reason_for_exception(SystemStallError("three role cycles"))
        == "SYSTEM_STALL"
    )
    assert _terminal_reason_for_exception(KeyboardInterrupt()) == "OPERATOR_CANCELLED"


def test_post_dispatch_model_failure_cannot_promote_safe_preliminary_state():
    preliminary = EvaluationReport(
        success=True,
        failure_type="none",
        metrics={"task_complete": 1.0, "payload_x_m": 1.0},
    )

    report = _post_dispatch_model_terminal_report(
        preliminary,
        error=ProviderFatalError("provider outage"),
        failed_subtask="coordinated_linear_transport",
        rollback_verification={"success": True},
    )

    assert not report.success
    assert report.failure_type == "manual_review_required"
    assert report.failure_level == "system"
    assert report.metrics["task_complete"] == 0.0
    terminal = report.evidence["post_execution_model_terminal"]
    assert terminal["physical_transaction_counted_once"] is True
    assert terminal["memory_promotion"] is False
    assert terminal["checkpoint_committed"] is False


def test_formal_protocol_disables_ancestor_rewind():
    assert not formal_allows_ancestor_rewind(FORMAL_EVAL_PROTOCOL_ID)
    assert formal_allows_ancestor_rewind("")
