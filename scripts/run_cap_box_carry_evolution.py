#!/usr/bin/env python3
"""Run the Code-as-Policy self-evolution loop for two-G1 box carry."""

from __future__ import annotations

import argparse
import ast
import atexit
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import inspect
from itertools import count
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Callable, TypeVar


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy import (  # noqa: E402
    AcceptedEpisodeManifest,
    BoxCarryEvaluator,
    BoxCarryTask,
    EvaluationReport,
    EvolutionController,
    EvolutionRecord,
    ExecutionTrace,
    G1SkillAPI,
    ObservationBundle,
    PolicyExecutor,
    PolicyScript,
    PlannedSkillCall,
    RobotAssignment,
    RobotExecutionTrace,
    RobotPolicyScript,
    RouteSpecConfigurationError,
    SimulatorCheckpointClient,
    CheckpointBackendFatalError,
    SkillCallRecord,
    Subtask,
    apply_transition_decision,
    transition_outcome_from_report,
    verify_transition,
    VisualVerifierReport,
    VerifiedActionMemoryStore,
    load_task_route_spec,
    validate_task_route_spec,
)
from praxis_coop.cap_policy.backends import (  # noqa: E402
    TRANSLATION_SKILLS,
    StaticSonicSkillBackend,
)
from praxis_coop.cap_policy.action_video import (  # noqa: E402
    PhysicalActionVideoRecorder,
)
from praxis_coop.cap_policy.evolution_agent import CollaborativeEvolutionAgent as TemplateEvolutionAgent  # noqa: E402
from praxis_coop.cap_policy.failure_taxonomy import (  # noqa: E402
    annotate_evaluation_report,
    classify_report,
)
from praxis_coop.cap_policy.evolution_controller import (  # noqa: E402
    EvolutionDecision,
    trace_has_physical_dispatch,
    trace_physical_transaction_completed,
)
from praxis_coop.cap_policy.model_agents import (  # noqa: E402
    ModelAgentError,
    ModelCollaborativeEvolutionAgent,
    ModelTaskOrchestrator,
    _active_branch_rewind_context,
    _body_frame_goal_alignment,
    _normalize_coordination_assignment_roles,
    _policy_motion_objective,
    _policy_transaction_for_prompt,
    _policy_transaction_signature,
)
from praxis_coop.cap_policy.local_navigation import (  # noqa: E402
    team_heading_deg,
    validate_strategy_against_local_navigation,
)
from praxis_coop.cap_policy.model_role_pipeline import (  # noqa: E402
    ModelRoleError,
    ModelRolePipeline,
    ModelRoleStallError,
)
from praxis_coop.cap_policy.generation_recovery import (  # noqa: E402
    GenerationStallTracker,
    HostContractError,
    MUTATION_SCOPE_INFEASIBLE,
    SystemStallError,
)
from praxis_coop.cap_policy.evaluator import (  # noqa: E402
    MAX_STRAIGHT_HEADING_ERROR_DEG,
    _active_goal_xy,
    _monitored_physical_failure,
    _route_regression_evidence,
)
from praxis_coop.cap_policy.orchestrator import TaskOrchestrator as TemplateTaskOrchestrator  # noqa: E402
from praxis_coop.cap_policy.policy_executor import validate_step_policy_code  # noqa: E402
from praxis_coop.cap_policy.scene_clearance import (  # noqa: E402
    pivot_candidate_swept_clearance,
    pivot_role_assignment,
)
from praxis_coop.cap_policy.result_review import build_run_review  # noqa: E402
from praxis_coop.cap_policy.simulation_diagnostics import (  # noqa: E402
    trusted_route_phase_guard,
)
from praxis_coop.cap_policy.visual_observation import ExternalCameraObserver  # noqa: E402
from praxis_coop.cap_policy.visual_verifier import (  # noqa: E402
    QwenVLBoxCarryVisualVerifier,
    make_qwen_vl_visual_verifier,
)
from praxis_coop.cap_policy.visualization import write_evolution_html  # noqa: E402
from praxis_coop.cap_policy.contracts import (  # noqa: E402
    VisualObservation,
    write_json,
)
from praxis_coop.cap_policy.coordination_specs import (  # noqa: E402
    COORDINATED_ARC_TRANSPORT,
    COORDINATED_LINEAR_TRANSPORT,
    COORDINATED_PIVOT,
    COORDINATION_PRIMITIVE_METHODS,
)
from praxis_coop.cap_policy.codebase_guard import (  # noqa: E402
    CodebaseGuard,
)
from praxis_coop.cap_policy.formal_eval import (  # noqa: E402
    FORMAL_BILLING_RECONCILIATION_TIMEOUT_S,
    FORMAL_EVAL_PROTOCOL_ID,
    FORMAL_TRANSACTION_JOURNAL_SCHEMA_VERSION,
    FORMAL_TRANSACTION_JOURNAL_STAGES,
    FormalEvalConfigurationError,
    configure_formal_args,
    formal_allows_ancestor_rewind,
    formal_task_overrides,
    validate_formal_freeze_commit,
    validate_formal_initial_runtime,
    validate_fresh_formal_run,
    write_formal_run_manifest,
)
from praxis_coop.cap_policy.strategy_memory import (  # noqa: E402
    StrategyContext,
    StrategyMemoryStore,
)
from praxis_coop.cap_policy.turn_episode import (  # noqa: E402
    TurnEpisodeError,
    empty_turn_episode,
    normalize_turn_episode,
    observe_passage_alignment,
    update_turn_episode,
    validate_policy_for_turn_episode,
    with_turn_episode,
)
from praxis_coop.sonic.runtime_state import (  # noqa: E402
    observe_payload_carry_stability,
    read_payload_runtime_status,
    wait_for_payload_initialization_ready,
    wait_for_payload_on_target,
    wait_for_payload_supported,
)
from praxis_coop.sonic.planner_keepalive import DualSonicIdleKeepalive  # noqa: E402
from praxis_coop.sonic.safety import (  # noqa: E402
    CARRY_SAFETY_PROFILES,
    get_carry_safety_profile,
)
from praxis_coop.configs import DEFAULT_MULTIMODAL_MODEL  # noqa: E402
from praxis_coop.llm import (  # noqa: E402
    OpenAICompatibleTextClient,
    ProviderFatalError,
    QwenVisionClient,
)
from praxis_coop.llm.billing_ledger import write_api_cost_ledger  # noqa: E402


SETTLED_CARRY_ACTIONS = frozenset(
    set(TRANSLATION_SKILLS) | {"turn_clockwise", "turn_counterclockwise"}
) | COORDINATION_PRIMITIVE_METHODS
STRAIGHT_TRANSLATION_ACTIONS = frozenset(TRANSLATION_SKILLS) | {
    COORDINATED_LINEAR_TRANSPORT
}
CHECKPOINT_ADVANCING_ACTIONS = frozenset(
    set(SETTLED_CARRY_ACTIONS)
    | {
        "place_payload_on_arms",
        "place_crouch",
        "recover_payload_inward",
        "recover_stand",
        "release_payload",
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
    }
)
CHECKPOINT_ROUNDTRIP_STABLE_S = 1.0
# A marginal loaded turn in the current scene remained apparently supported
# for about 28 seconds before the payload rolled and one robot fell. Keep the
# candidate provisional long enough to expose that delayed failure before it
# enters the accepted manifest or executable memory.
LOADED_TURN_CHECKPOINT_STABLE_S = 30.0
LOADED_TURN_ACTIONS = frozenset(
    {
        "turn_clockwise",
        "turn_counterclockwise",
        COORDINATED_ARC_TRANSPORT,
        COORDINATED_PIVOT,
    }
)
POST_RELEASE_CHECKPOINT_ACTIONS = frozenset(
    {
        "release_payload",
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
        "recover_stand",
    }
)
CHECKPOINT_LOCAL_EXHAUSTION_MARKERS = (
    "no legal coordination primitive objective remains after physical budget exclusions",
    "checkpoint-local coordination candidate exclusions exhausted",
    "all coordination modes for api.",
    "no model-selectable sdk methods remain after verifier exclusions",
    "verifier-rejected executable transaction",
    "repeats the exact verifier-rejected executable ast",
    "do not repeat api.",
    "structured coordination candidate search exhausted",
    "structured measured-sync candidate search exhausted",
)
ROLLBACK_POSE_LIMITS = {
    "max_payload_xy_error_m": 0.06,
    "max_robot_xy_error_m": 0.06,
    "max_z_error_m": 0.03,
    # Under a free carried payload, the independent planner reset can settle a
    # root by just over five degrees even when payload pose, dual support, and
    # both robot positions return correctly. This remains a rollback-fidelity
    # gate; route heading is evaluated separately from the live state.
    "max_yaw_error_deg": 6.0,
}
POST_RESTORE_HEADING_CORRECTION_LIMIT = 3
_ACTIVE_RUN_FINALIZER: dict[str, object] | None = None


class TurnReadinessError(TurnEpisodeError):
    """One exact model-authored pivot lacks live full-team clearance."""

    def __init__(self, message: str, *, evidence: dict[str, object]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


class CandidateCheckpointRejectedError(RuntimeError):
    """An uncommitted candidate failed restore/stability; its parent may remain valid."""


def _raise_checkpoint_restore_drift(
    *,
    candidate_checkpoint: bool,
    verification_path: Path,
) -> None:
    """Classify candidate drift separately from trusted-parent restore failure."""

    error = (
        "checkpoint rollback drift exceeded the formal experiment tolerance; "
        f"see {verification_path}"
    )
    if candidate_checkpoint:
        raise CandidateCheckpointRejectedError(error)
    raise CheckpointBackendFatalError(error)


def _checkpoint_roundtrip_stable_s(policy_action: str) -> float:
    """Give loaded turns enough time to expose delayed payload slip."""

    if policy_action in LOADED_TURN_ACTIONS:
        return LOADED_TURN_CHECKPOINT_STABLE_S
    return CHECKPOINT_ROUNDTRIP_STABLE_S


def _checkpoint_authoritative_planning_state(
    *,
    live_runtime_state: dict,
    checkpoint_reference_state: dict,
    turn_episode: dict,
) -> dict:
    """Keep model planning anchored to the state that the next action restores."""

    authoritative_state = (
        checkpoint_reference_state
        if isinstance(checkpoint_reference_state, dict) and checkpoint_reference_state
        else live_runtime_state
    )
    return with_turn_episode(authoritative_state, turn_episode)


def _checkpoint_reference_after_ancestor_rewind(
    rewound_state: dict[str, object],
) -> dict[str, object]:
    """Retain the complete restored state, never its compact pose annotation."""

    if not isinstance(rewound_state, dict) or not rewound_state:
        raise CheckpointBackendFatalError(
            "automatic ancestor rewind produced no authoritative runtime state"
        )
    return rewound_state


@contextmanager
def _paused_idle_keepalive(
    keepalive: DualSonicIdleKeepalive | None,
):
    """Release planner ports for one bounded control-plane operation."""

    if keepalive is None:
        yield
        return

    keepalive.stop()
    try:
        yield
    finally:
        keepalive.start()


@contextmanager
def _checkpoint_heartbeat_lease(
    keepalive: DualSonicIdleKeepalive | None,
):
    """Keep LowCmd alive while reserving the heartbeat file for checkpoint I/O."""

    if keepalive is None:
        yield
        return

    keepalive.start()
    with keepalive.suspend_heartbeat_writes():
        yield


def _publish_experiment_status(
    role_pipeline: ModelRolePipeline | None,
    runtime_status: str,
    *,
    attempt: int,
    checkpoint_id: str,
    policy: PolicyScript | None = None,
    report: EvaluationReport | None = None,
    trace: ExecutionTrace | None = None,
    checkpoint_before: str = "",
    checkpoint_after: str = "",
    rollback_performed: bool = False,
    next_system_operation: str = "",
    artifact_dir: str | Path = "",
    details: dict | None = None,
) -> dict:
    """Publish one runner transition without affecting template-mode ablations."""

    if role_pipeline is None:
        return {}
    last_outcome = None
    if report is not None:
        checkpoint_outcome = report.evidence.get("checkpoint_outcome", {})
        checkpoint_outcome = (
            checkpoint_outcome if isinstance(checkpoint_outcome, dict) else {}
        )
        last_outcome = {
            "status": "accepted" if report.success else "rejected",
            "verifier_success": bool(report.success),
            "failure_type": report.failure_type,
            "failure_level": classify_report(report).level,
            "diagnosis": report.diagnosis,
            "recommendations": list(report.recommendations),
            "task_complete": bool(report.metrics.get("task_complete", 0.0)),
            "execution_status": trace.status if trace is not None else "",
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "rollback_performed": bool(rollback_performed),
            "checkpoint_outcome": checkpoint_outcome,
            "transition_outcome": transition_outcome_from_report(report).value,
        }
    return role_pipeline.publish_high_level_status(
        runtime_status,
        attempt=attempt,
        checkpoint_id=checkpoint_id,
        policy=asdict(policy) if policy is not None else None,
        last_outcome=last_outcome,
        next_system_operation=next_system_operation,
        artifact_dir=artifact_dir,
        details=details,
    )


def _run_main() -> None:
    global _ACTIVE_RUN_FINALIZER

    args = parse_args()
    host_contract_protocol = (
        args.eval_protocol
        or (
            FORMAL_EVAL_PROTOCOL_ID
            if args.diagnostic_formal_host_contracts
            else ""
        )
    )
    validate_fresh_formal_run(args)
    validate_formal_freeze_commit(
        args,
        repo_root=REPO_ROOT,
        codebase_snapshot={},
    )
    validate_formal_initial_runtime(
        args,
        read_json(args.runtime_state_file),
    )
    if args.agent_mode == "model":
        if args.visual_evaluator != "model":
            raise RuntimeError(
                "the model self-evolution path requires the multimodal evaluator"
            )
        if (
            args.planner_backend == "local_qwen_vl"
            and args.vl_model_name != DEFAULT_MULTIMODAL_MODEL
        ):
            raise RuntimeError(
                "the local formal backend requires Task Orchestrator, Coder, "
                "VLM Evaluator/Failure Analyzer, Evolution Manager, Mutation, "
                "Code Repair, and Memory Update to share one "
                f"{DEFAULT_MULTIMODAL_MODEL} instance"
            )
        if (
            args.planner_backend == "openai_compatible"
            and not os.environ.get(args.planner_api_key_env, "").strip()
        ):
            raise RuntimeError(
                "remote multimodal backend requested but credential environment "
                f"variable {args.planner_api_key_env} is not configured"
            )
    safety = get_carry_safety_profile(args.physical_safety_profile)
    task = BoxCarryTask(
        task_id=args.task_id,
        target=args.target,
        task_stage=args.task_stage,
        route_discovery=args.route_discovery,
        min_payload_z_m=safety.min_payload_z_m,
        min_robot_pelvis_height_m=safety.min_robot_pelvis_height_m,
        min_robot_xy_separation_m=safety.min_robot_xy_separation_m,
        require_payload_on_target=not args.no_require_payload_on_target,
        require_robots_exit=args.require_robots_exit,
        **formal_task_overrides(host_contract_protocol),
    )
    try:
        if args.route_spec is not None:
            task = load_task_route_spec(task, args.route_spec)
        route_spec_validation = validate_task_route_spec(
            task,
            source=str(args.route_spec or "BoxCarryTask defaults"),
        )
    except RouteSpecConfigurationError as exc:
        raise FormalEvalConfigurationError(
            f"SYSTEM_INVALID/configuration_error: {exc}"
        ) from exc
    experiment_started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "route_spec_validation.json", route_spec_validation)
    codebase_guard = CodebaseGuard.initialize(
        repo_root=REPO_ROOT,
        output_dir=args.output_dir,
        require_clean=(
            args.agent_mode == "model"
            and args.execute
            and not args.allow_dirty_codebase
        ),
    )
    strategy_store = StrategyMemoryStore(args.strategy_memory)
    if not args.strategy_memory.exists():
        strategy_store.save()
    action_memory = VerifiedActionMemoryStore(args.action_exemplar_memory)
    if not args.action_exemplar_memory.exists():
        action_memory.save()
    episode_manifest = AcceptedEpisodeManifest(args.episode_manifest)
    resume_starts_new_episode = bool(
        args.resume_checkpoint_id is not None
        and not episode_manifest.latest_checkpoint
    )
    agent_client = None
    write_formal_run_manifest(
        args=args,
        repo_root=REPO_ROOT,
        codebase_snapshot=codebase_guard.snapshot,
        task=task,
        status="initialized",
    )
    backend = StaticSonicSkillBackend(
        execute=args.execute,
        default_start_pulse_seconds=args.start_pulse_seconds,
        default_stand_seconds=args.stand_seconds,
        minimum_transport_window_s=args.minimum_transport_window_s,
        minimum_lateral_transport_window_s=args.minimum_lateral_transport_window_s,
        nominal_distance_duration_gain=args.transport_duration_gain,
        nominal_distance_duration_baseline_m=args.transport_duration_baseline_m,
        minimum_transport_ramp_s=args.minimum_transport_ramp_s,
        post_transport_idle_s=args.post_transport_idle_s,
        payload_safety_profile=args.physical_safety_profile,
        team_sync_enabled=not args.no_team_sync,
        team_sync_active_catchup=args.team_sync_active_catchup,
        team_sync_longitudinal_kp=args.team_sync_longitudinal_kp,
        team_sync_cross_track_kp=args.team_sync_cross_track_kp,
        team_sync_payload_kp=args.team_sync_payload_kp,
        team_sync_error_filter_alpha=args.team_sync_error_filter_alpha,
        team_sync_pause_leader_error_m=args.team_sync_pause_leader_error_m,
        team_sync_resume_error_m=args.team_sync_resume_error_m,
        team_sync_lagging_boost_mps=args.team_sync_lagging_boost_mps,
        team_sync_max_correction_mps=args.team_sync_max_correction_mps,
        team_sync_min_active_speed_ratio=args.team_sync_min_active_speed_ratio,
        team_sync_hard_desync_m=(
            safety.team_hard_desync_m
            if safety.team_hard_desync_m is not None
            else safety.max_support_center_error_m
        ),
        linear_response_calibration_file=(
            args.output_dir / "motion_calibration" / "linear_response.json"
        ),
    )
    idle_keepalive = (
        DualSonicIdleKeepalive(
            heartbeat_file=args.heartbeat_file,
            front_port=backend.front_port,
            back_port=backend.back_port,
            publish_hz=backend.publish_hz,
        )
        if args.execute
        else None
    )
    if idle_keepalive is not None:
        atexit.register(idle_keepalive.stop)
    checkpoint_client = None
    checkpoint_reference_state: dict = {}
    accepted_steps = _initial_accepted_step_counter(
        args.resume_checkpoint_id,
        next_accepted_step_index=args.next_accepted_step_index,
    )
    checkpoint_id = args.resume_checkpoint_id or "accepted_step_00"
    experiment_initial_state = read_json(args.runtime_state_file)
    explicit_retired_child_signature = ""
    explicit_retired_child_transaction = ""
    explicit_retired_child_motion_objective = ""
    if args.execute:
        startup_state = read_json(args.runtime_state_file)
        if args.route_discovery:
            _require_route_discovery_global_camera(startup_state)
        checkpoint_client = SimulatorCheckpointClient(
            heartbeat_file=args.heartbeat_file,
            runtime_state_file=args.runtime_state_file,
            artifact_dir=args.output_dir / "checkpoint_events",
            timeout_s=args.checkpoint_timeout_s,
            restore_settle_s=args.checkpoint_resync_settle_s,
        )
        if args.resume_checkpoint_id is not None:
            available_ids = startup_state.get("checkpoint", {}).get("available_ids", [])
            if checkpoint_id not in available_ids:
                raise RuntimeError(
                    f"requested checkpoint {checkpoint_id!r} is not available in the live "
                    f"Term1 runtime; available={available_ids}"
                )
            with _checkpoint_heartbeat_lease(idle_keepalive):
                initial_restored_checkpoint = checkpoint_client.restore(
                    checkpoint_id,
                    hold_planning=True,
                )
            # Term1 freezes physics atomically with the restore event. Reset both
            # planners only after that point so the rejected endpoint never keeps
            # walking while rollback is in progress.
            with _paused_idle_keepalive(idle_keepalive):
                heading_resync = backend.resynchronize_idle(
                    output_dir=args.output_dir / "resume_checkpoint_heading_resync",
                    heartbeat_file=args.heartbeat_file,
                    runtime_state_file=args.runtime_state_file,
                    duration_s=max(0.5, float(args.checkpoint_resync_idle_s)),
                    stand_phase_id="resume_checkpoint_heading_resync",
                )
            # The planner reset consumes the exact restored LowState while physics
            # is frozen. Reassert qpos/qvel once more afterwards so the first live
            # integration tick starts from the checkpoint, not from planner-history
            # mismatch accumulated before the reset.
            with _checkpoint_heartbeat_lease(idle_keepalive):
                restored_checkpoint = checkpoint_client.restore(
                    checkpoint_id,
                    hold_planning=True,
                )
            _wait_for_checkpoint_resync_complete(
                args.runtime_state_file,
                checkpoint_id=checkpoint_id,
                timeout_s=max(
                    args.checkpoint_timeout_s,
                    args.checkpoint_resync_settle_s + 4.0,
                ),
            )
            initial_support = _wait_for_initial_physical_state(
                runtime_state_file=args.runtime_state_file,
                timeout_s=args.initial_support_timeout_s,
                stable_s=args.initial_support_stable_s,
                label=f"resume checkpoint {checkpoint_id}",
                safety_profile=args.physical_safety_profile,
            )
            checkpoint_reference_state = (
                restored_checkpoint.runtime_state_after or read_json(args.runtime_state_file)
            )
            experiment_initial_state = read_json(args.runtime_state_file)
            write_json(
                args.output_dir / "checkpoint_resume_source.json",
                {
                    "checkpoint_id": checkpoint_id,
                    "accepted_step_count": accepted_steps,
                    "available_ids": available_ids,
                    "restore_event_id": restored_checkpoint.event_id,
                    "initial_restore_event_id": initial_restored_checkpoint.event_id,
                    "reassert_restore_event_id": restored_checkpoint.event_id,
                    "heading_resync": heading_resync,
                    "rollback_order": (
                        "restore_freeze_then_dual_planner_reset_then_exact_reassert_then_release"
                    ),
                },
            )
            print(
                f"[cap-box-carry] resumed live checkpoint={checkpoint_id}; "
                "dual planner histories reset before model inference",
                flush=True,
            )
        else:
            if _runtime_has_verified_payload_support(
                startup_state,
                safety_profile=args.physical_safety_profile,
            ):
                bootstrap = {
                    "status": "skipped_already_initialized",
                    "reason": (
                        "fresh live state already has healthy SONIC control and continuous "
                        "front/back hand-or-forearm payload support"
                    ),
                }
                print(
                    "[cap-box-carry] bootstrap START/IDLE skipped: verified payload "
                    "initialization is already live",
                    flush=True,
                )
            else:
                bootstrap = backend.resynchronize_idle(
                    output_dir=args.output_dir / "bootstrap_idle",
                    heartbeat_file=args.heartbeat_file,
                    runtime_state_file=args.runtime_state_file,
                    duration_s=args.bootstrap_idle_seconds,
                    start_pulse_seconds=args.bootstrap_start_pulse_seconds,
                    stand_phase_id="static_skill_pre_stand",
                )
            write_json(args.output_dir / "bootstrap_idle.json", bootstrap)
            initial_support = _wait_for_initial_physical_state(
                runtime_state_file=args.runtime_state_file,
                timeout_s=args.initial_support_timeout_s,
                stable_s=args.initial_support_stable_s,
                label="startup",
                safety_profile=args.physical_safety_profile,
            )
            write_json(
                args.output_dir / "initial_physical_gate.json",
                _payload_status_payload(initial_support),
            )
            with _checkpoint_heartbeat_lease(idle_keepalive):
                saved_checkpoint = checkpoint_client.save(checkpoint_id, reset_metrics=True)
            with _checkpoint_heartbeat_lease(idle_keepalive):
                checkpoint_client.restore(checkpoint_id, hold_planning=True)
            _wait_for_checkpoint_resync_complete(
                args.runtime_state_file,
                checkpoint_id=checkpoint_id,
                timeout_s=max(
                    args.checkpoint_timeout_s,
                    args.checkpoint_resync_settle_s + 4.0,
                ),
            )
            checkpoint_reference_state = saved_checkpoint.runtime_state_after or read_json(
                args.runtime_state_file
            )
            experiment_initial_state = checkpoint_reference_state
            print(
                f"[cap-box-carry] committed initial checkpoint={checkpoint_id} before model inference",
                flush=True,
            )
            write_json(
                args.output_dir / "initial_checkpoint_physical_gate.json",
                _payload_status_payload(initial_support),
            )
        idle_keepalive.start()
        print(
            "[cap-box-carry] dual SONIC IDLE keepalive active during camera/VLM/LLM planning",
            flush=True,
        )
        if (
            args.resume_checkpoint_id is not None
            and episode_manifest.latest_checkpoint
            and episode_manifest.latest_checkpoint != checkpoint_id
        ):
            retired_child_policy = _accepted_child_policy_after_checkpoint(
                episode_manifest.data,
                checkpoint_id=checkpoint_id,
            )
            if retired_child_policy is not None:
                explicit_retired_child_signature = _policy_transaction_signature(
                    retired_child_policy
                )
                explicit_retired_child_transaction = _policy_transaction_for_prompt(
                    retired_child_policy
                )
                retired_method, retired_objective = _policy_motion_objective(
                    retired_child_policy
                )
                explicit_retired_child_motion_objective = (
                    f"{retired_method}:{retired_objective}"
                    if retired_objective
                    else retired_method
                )
            retired_source_artifacts = _manifest_source_artifacts_after_checkpoint(
                episode_manifest.data,
                checkpoint_id,
            )
            retired_episode_steps = episode_manifest.rewind_to_checkpoint(checkpoint_id)
            retired_action_exemplars = action_memory.revoke_source_artifacts(
                retired_source_artifacts
            )
            write_json(
                args.output_dir / "episode_manifest_resume_rewind.json",
                {
                    "restored_checkpoint_id": checkpoint_id,
                    "retired_episode_steps": retired_episode_steps,
                    "retired_action_exemplars": retired_action_exemplars,
                    "retired_source_artifacts": list(retired_source_artifacts),
                    "retired_child_transaction_signature": (
                        explicit_retired_child_signature
                    ),
                    "retired_child_transaction": (
                        explicit_retired_child_transaction
                    ),
                    "reason": "explicit live checkpoint resume",
                },
            )
            print(
                "[cap-box-carry] accepted episode manifest rewound to explicit "
                f"resume checkpoint={checkpoint_id}; retired_steps={retired_episode_steps} "
                f"retired_action_exemplars={retired_action_exemplars}",
                flush=True,
            )
        episode_manifest.initialize(
            checkpoint_id=checkpoint_id,
            initial_state=experiment_initial_state,
            reset=args.resume_checkpoint_id is None,
        )
    straight_heading_ledger_path = (
        args.runtime_state_file.parent / "straight_heading_references.json"
    )
    straight_heading_references = _load_straight_heading_references(
        straight_heading_ledger_path,
        reset=args.resume_checkpoint_id is None,
    )
    straight_heading_reference_rad = _straight_heading_reference_for_checkpoint(
        checkpoint_id,
        references=straight_heading_references,
        fallback_state=experiment_initial_state,
        runtime_state_file=args.runtime_state_file,
        resume_as_episode_root=resume_starts_new_episode,
    )
    straight_heading_references[checkpoint_id] = straight_heading_reference_rad
    _write_straight_heading_references(
        straight_heading_ledger_path,
        active_checkpoint_id=checkpoint_id,
        references=straight_heading_references,
    )
    if args.execute and args.resume_checkpoint_id is not None:
        if resume_starts_new_episode:
            startup_heading_normalization = _preserve_resume_root_heading_report(
                read_json(args.runtime_state_file),
                reference_yaw_rad=straight_heading_reference_rad,
            )
            write_json(
                args.output_dir
                / "resume_checkpoint_heading_normalization"
                / "heading_normalization.json",
                startup_heading_normalization,
            )
        else:
            startup_heading_normalization = _normalize_straight_heading_after_restore(
                backend=backend,
                heartbeat_file=args.heartbeat_file,
                runtime_state_file=args.runtime_state_file,
                output_dir=args.output_dir / "resume_checkpoint_heading_normalization",
                idle_keepalive=idle_keepalive,
                reference_yaw_rad=straight_heading_reference_rad,
                safety_profile=args.physical_safety_profile,
            )
        write_json(
            args.output_dir / "resume_checkpoint_heading_normalization.json",
            startup_heading_normalization,
        )
    resume_attempt_dir = args.resume_attempt_dir or args.resume_failed_attempt_dir
    resume_context_dir = (
        resume_attempt_dir.parent
        if resume_attempt_dir is not None
        else (
            args.resume_policy.parent.parent
            if args.resume_policy is not None
            and args.resume_policy.parent.name.startswith("attempt_")
            else args.resume_policy.parent
            if args.resume_policy is not None
            else None
        )
    )
    resume_history_dirs = _resume_history_output_dirs(resume_attempt_dir)
    model_evidence_history_dirs = _resume_model_evidence_history_dirs(
        resume_attempt_dir,
        resume_history_dirs=resume_history_dirs,
        checkpoint_id=str(args.resume_checkpoint_id or ""),
        accepted_manifest=episode_manifest.data,
    )
    controller_history_dirs = (
        (resume_attempt_dir.parent,)
        if resume_attempt_dir is not None
        else ()
    )
    evolution_controller = EvolutionController(
        max_physical_iterations=args.iteration_budget,
        strategy_failure_budget=args.strategy_failure_budget,
        output_dir=args.output_dir,
        history_dirs=controller_history_dirs,
    )
    resume_checkpoint_turn_episode = (
        _reconstruct_turn_episode_from_manifest(
            episode_manifest.data,
            checkpoint_id=str(args.resume_checkpoint_id),
        )
        if args.resume_checkpoint_id is not None
        else {}
    )
    branch_rewind_context: dict[str, object] | None = None
    inherited_branch_rewind_context = (
        read_json(resume_context_dir / "branch_rewind_context.json")
        if resume_context_dir is not None
        else {}
    )
    inherited_rewind_scope = str(
        inherited_branch_rewind_context.get("scope_checkpoint_id", "")
    ).strip()
    inherited_turn_episode = (
        read_json(resume_context_dir / "turn_episode.json")
        if resume_context_dir is not None
        else {}
    )
    resume_checkpoint_turn_episode = _select_resume_turn_episode(
        reconstructed=resume_checkpoint_turn_episode,
        inherited=inherited_turn_episode,
        checkpoint_id=str(args.resume_checkpoint_id or ""),
        resume_failed_attempt=args.resume_failed_attempt_dir is not None,
    )
    inherited_route_rewind_pending = bool(
        inherited_branch_rewind_context.get("persistence_mode")
        == "until_route_turn_probe_success"
        and not (
            str(
                resume_checkpoint_turn_episode.get(
                    "purpose",
                    inherited_turn_episode.get("purpose", ""),
                )
            )
            == "route_turn"
            and str(
                resume_checkpoint_turn_episode.get(
                    "status",
                    inherited_turn_episode.get("status", ""),
                )
            )
            == "completed"
            and not bool(
                resume_checkpoint_turn_episode.get(
                    "active",
                    inherited_turn_episode.get("active", False),
                )
            )
        )
    )
    inherited_runtime_state = dict(experiment_initial_state)
    inherited_runtime_checkpoint = inherited_runtime_state.get("checkpoint", {})
    inherited_runtime_checkpoint = (
        dict(inherited_runtime_checkpoint)
        if isinstance(inherited_runtime_checkpoint, dict)
        else {}
    )
    inherited_runtime_checkpoint.setdefault(
        "checkpoint_id", str(args.resume_checkpoint_id or "").strip()
    )
    inherited_runtime_state["checkpoint"] = inherited_runtime_checkpoint
    inherited_rewind_active = bool(
        inherited_branch_rewind_context.get("active", False)
        and (
            _active_branch_rewind_context(
                {"branch_rewind_context": inherited_branch_rewind_context},
                inherited_runtime_state,
            )
            or inherited_route_rewind_pending
        )
    )
    if args.branch_rewind_note:
        explicit_route_exclusion = str(
            args.branch_rewind_route_exclusion or ""
        ).strip()
        branch_rewind_context = _build_branch_rewind_context(
            task=task,
            checkpoint_id=str(args.resume_checkpoint_id),
            lesson=str(args.branch_rewind_note),
            runtime_state=experiment_initial_state,
            blocked_motion_objective=(
                explicit_route_exclusion
                or explicit_retired_child_motion_objective
                or (
                    str(
                        inherited_branch_rewind_context.get(
                            "blocked_motion_objective",
                            "",
                        )
                    )
                    if (
                        inherited_rewind_active
                        and inherited_rewind_scope
                        == str(args.resume_checkpoint_id or "").strip()
                    )
                    else ""
                )
            ),
            enforce_blocked_motion_objective=bool(explicit_route_exclusion),
            blocked_primitive_families=(),
            retired_child_transaction_signature=(
                explicit_retired_child_signature
            ),
            retired_child_transaction=explicit_retired_child_transaction,
            source="operator_branch_rewind_note",
        )
        write_json(args.output_dir / "branch_rewind_context.json", branch_rewind_context)
    elif inherited_rewind_active:
        branch_rewind_context = dict(inherited_branch_rewind_context)
        if not bool(
            branch_rewind_context.get("enforce_blocked_motion_objective", False)
        ):
            branch_rewind_context.update(
                {
                    "blocked_motion_objective": "",
                    "blocked_primitive_families": [],
                }
            )
        write_json(args.output_dir / "branch_rewind_context.json", branch_rewind_context)
    if args.resume_checkpoint_id is not None:
        retired_failure_count = (
            evolution_controller.rewind_strategy_evidence_to_checkpoint(
                args.resume_checkpoint_id,
                release_checkpoint_failures=bool(args.branch_rewind_note),
            )
        )
        write_json(
            args.output_dir / "resume_controller_evidence_scope.json",
            {
                "checkpoint_id": args.resume_checkpoint_id,
                "retired_child_branch_failures": retired_failure_count,
                "physical_iterations_preserved": evolution_controller.physical_iterations,
            },
        )
    shared_model_client = None
    role_pipeline = None
    if args.agent_mode == "model":
        task_plan_resume_dir = (
            resume_attempt_dir
            if resume_attempt_dir is not None
            else args.resume_policy.parent
            if args.resume_policy is not None
            and args.resume_policy.parent.name.startswith("attempt_")
            else None
        )
        resumed_task_plan, resumed_task_plan_path = (
            _resume_task_plan_from_attempt(task_plan_resume_dir)
        )
        persisted_plan_path = (
            args.output_dir
            / "model_roles"
            / "task_orchestrator_plan_latest.json"
        )
        active_plan_path = (
            args.output_dir / "model_roles" / "active_task_plan_latest.json"
        )
        if resumed_task_plan and (
            args.resume_checkpoint_id is not None
            or (
                not persisted_plan_path.exists()
                and not active_plan_path.exists()
            )
        ):
            superseded_plan = (
                read_json(active_plan_path)
                or read_json(persisted_plan_path)
            )
            write_json(persisted_plan_path, resumed_task_plan)
            write_json(active_plan_path, resumed_task_plan)
            write_json(
                args.output_dir / "resume_task_plan_source.json",
                {
                    "resume_attempt_dir": str(task_plan_resume_dir),
                    "source_task_plan": str(resumed_task_plan_path),
                    "explicit_checkpoint_rewind": (
                        args.resume_checkpoint_id is not None
                    ),
                    "superseded_descendant_plan": superseded_plan,
                    "policy": (
                        "seed the persistent Orchestrator plan from the exact resume "
                        "attempt after checkpoint rewind, then update it from current "
                        "trusted physics"
                    ),
                },
            )
        if args.planner_backend == "local_qwen_vl":
            shared_model_client = QwenVisionClient(
                model_name=args.vl_model_name,
                device_map=args.vl_device_map,
                max_gpu_memory=args.vl_max_gpu_memory,
                max_cpu_memory=args.vl_max_cpu_memory,
                quantization=args.vl_quantization,
                temperature=args.vl_temperature,
                max_new_tokens=args.vl_max_new_tokens,
                text_temperature=args.temperature,
                text_max_new_tokens=args.max_new_tokens,
            )
            if args.preload_vl_model:
                shared_model_client.preload()
            role_model_name = args.vl_model_name
        else:
            shared_model_client = OpenAICompatibleTextClient(
                model_name=args.planner_model,
                base_url=args.planner_base_url,
                api_key_env=args.planner_api_key_env,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                timeout_s=args.planner_timeout_s,
                network_retries=args.planner_network_retries,
                seed=args.seed,
                keep_alive_on_transient_failure=(
                    args.keep_term4_on_transient_provider_error
                ),
                transient_recovery_backoff_s=args.provider_recovery_backoff_s,
                transient_recovery_wall_time_cap_s=(
                    args.provider_recovery_wall_time_cap_s
                ),
                required_resolved_model=(
                    args.planner_model if host_contract_protocol else ""
                ),
            )
            role_model_name = args.planner_model
        agent_client = shared_model_client
        role_pipeline = ModelRolePipeline(
            client=shared_model_client,
            output_dir=args.output_dir,
            model_name=role_model_name,
            require_formal_model=args.planner_backend == "local_qwen_vl",
        )
        _publish_experiment_status(
            role_pipeline,
            "initializing",
            attempt=args.next_policy_attempt_index or 0,
            checkpoint_id=checkpoint_id,
            next_system_operation=(
                "capture the trusted global view, then ask Task Orchestrator "
                "for the current phase and subtask"
            ),
            artifact_dir=args.output_dir,
        )
        orchestrator = ModelTaskOrchestrator(
            client=agent_client,
            default_transport_skill=args.transport_skill,
            default_transport_distance_m=args.transport_distance_m,
            default_transport_speed_mps=args.transport_speed_mps,
            default_transport_ramp_s=args.transport_ramp_s,
            task_stage=args.task_stage,
            route_discovery=args.route_discovery,
            coordination_layer_enabled=True,
            action_memory=action_memory,
            role_pipeline=role_pipeline,
            accepted_episode_manifest_path=args.episode_manifest,
            formal_eval=bool(host_contract_protocol),
        )
        evolver = ModelCollaborativeEvolutionAgent(
            args.output_dir,
            client=agent_client,
            task=task,
            task_stage=args.task_stage,
            route_discovery=args.route_discovery,
            strategy_failure_budget=args.strategy_failure_budget,
            history_dirs=model_evidence_history_dirs,
            checkpoint_relation_history_dirs=resume_history_dirs,
            evolution_controller=evolution_controller,
            coordination_layer_enabled=True,
            action_memory=action_memory,
            accepted_episode_manifest_path=args.episode_manifest,
            branch_rewind_context=branch_rewind_context,
            role_pipeline=role_pipeline,
            formal_eval=bool(host_contract_protocol),
        )
    else:
        orchestrator = TemplateTaskOrchestrator(
            default_transport_skill=args.transport_skill,
            default_transport_distance_m=args.transport_distance_m,
            default_transport_speed_mps=args.transport_speed_mps,
            task_stage=args.task_stage,
        )
        evolver = TemplateEvolutionAgent(args.output_dir)
    camera = ExternalCameraObserver(
        output_dir=args.output_dir / "visual_observations",
        camera_device=args.camera_device,
        frame_path=args.visual_frame,
        window_name=args.visual_window_name,
    )
    global_observation = experiment_initial_state.get("global_observation", {})
    global_observation = (
        global_observation if isinstance(global_observation, dict) else {}
    )
    action_video_source = Path(
        str(global_observation.get("frame_path", "")).strip()
        or str(args.runtime_state_file.with_name("vlm_global_frame.png"))
    )
    action_video_recorder = PhysicalActionVideoRecorder(
        source_frame_path=action_video_source,
        output_dir=args.output_dir / "action_videos",
        fps=args.action_video_fps,
        enabled=bool(args.record_action_videos and args.execute),
    )
    visual_verifier = None
    if args.visual_evaluator == "model":
        if shared_model_client is not None:
            visual_verifier = QwenVLBoxCarryVisualVerifier(
                client=shared_model_client,
                model_name=role_model_name,
                role_pipeline=role_pipeline,
            )
        else:
            visual_verifier = make_qwen_vl_visual_verifier(
                model_name=args.vl_model_name,
                device_map=args.vl_device_map,
                max_gpu_memory=args.vl_max_gpu_memory,
                max_cpu_memory=args.vl_max_cpu_memory,
                quantization=args.vl_quantization,
                temperature=args.vl_temperature,
                max_new_tokens=args.vl_max_new_tokens,
            )
        if args.preload_vl_model and shared_model_client is None:
            visual_verifier.preload()
    evaluator = BoxCarryEvaluator(
        args.runtime_state_file,
        safety_profile=args.physical_safety_profile,
        require_visual=args.visual_evaluator == "model",
    )

    records: list[EvolutionRecord] = []
    automatic_branch_rewinds: list[dict[str, object]] = []
    generation_stalls = GenerationStallTracker(args.output_dir)
    finalization_state = {"completed": False}

    def finalize_run(
        *,
        status: str = "",
        terminal_error: BaseException | None = None,
    ) -> None:
        if finalization_state["completed"]:
            return
        video_timeline = action_video_recorder.finalize(
            run_status=str(status).strip() or "terminalizing"
        )
        result = _finalize_experiment(
            args=args,
            task=task,
            records=records,
            experiment_initial_state=experiment_initial_state,
            experiment_started_at=experiment_started_at,
            codebase_guard=codebase_guard,
            agent_client=agent_client,
            visual_verifier=visual_verifier,
            role_pipeline=role_pipeline,
            evolution_controller=evolution_controller,
            status=status,
            terminal_error=terminal_error,
        )
        if video_timeline.get("enabled", True):
            video_timeline["run_status"] = str(result.get("status", status))
            write_json(
                args.output_dir / "action_videos" / "timeline.json",
                video_timeline,
            )
        finalization_state["completed"] = True

    _ACTIVE_RUN_FINALIZER = {"finalize": finalize_run}
    turn_episode_path = args.output_dir / "turn_episode.json"
    reconstructed_turn_episode = resume_checkpoint_turn_episode
    if args.resume_checkpoint_id is not None and args.branch_rewind_note:
        turn_episode = _reopen_turn_episode_after_branch_rewind(
            reconstructed_turn_episode or empty_turn_episode(),
            checkpoint_id=str(args.resume_checkpoint_id),
            reason=str(args.branch_rewind_note),
            state_source=(
                "accepted_episode_manifest_replay"
                if reconstructed_turn_episode
                else "empty_after_branch_rewind"
            ),
        )
    else:
        turn_episode = normalize_turn_episode(
            reconstructed_turn_episode
            if args.resume_checkpoint_id is not None and reconstructed_turn_episode
            else read_json(turn_episode_path)
            if args.resume_checkpoint_id is not None and turn_episode_path.exists()
            else inherited_turn_episode
            if args.resume_checkpoint_id is not None and inherited_turn_episode
            else empty_turn_episode()
        )
    write_json(turn_episode_path, turn_episode)
    checkpoint_lineage_dirs = tuple(
        dict.fromkeys((*resume_history_dirs, args.output_dir))
    )
    stop_before_loop = False
    live_initial_runtime_state = with_turn_episode(
        read_json(args.runtime_state_file), turn_episode
    )
    initial_runtime_state = _checkpoint_authoritative_planning_state(
        live_runtime_state=live_initial_runtime_state,
        checkpoint_reference_state=checkpoint_reference_state,
        turn_episode=turn_episode,
    )
    initial_visual = camera.observe(
        label="initial_observation",
        runtime_state=live_initial_runtime_state,
    )
    if args.agent_mode == "model" and args.visual_evaluator == "model":
        _require_visual_frame(initial_visual, label="initial model observation")
    write_json(
        args.output_dir / "initial_observation.json",
        {
            "frame_path": initial_visual.frame_path,
            "checkpoint_id": checkpoint_id,
            "model_role_invoked": False,
            "consumer": "task_orchestrator",
            "reason": (
                "the Evaluator/Failure Analyzer is a post-execution role; the "
                "Orchestrator receives the initial global frame directly"
            ),
        },
    )
    observation = ObservationBundle(
        runtime_state_path=str(args.runtime_state_file),
        runtime_state=initial_runtime_state,
        visual_observations=(initial_visual,),
        retrieved_specs=tuple(
            strategy_store.retrieve(
                StrategyContext(
                    phase=_strategy_phase(
                        task,
                        initial_runtime_state,
                        accepted_steps=episode_manifest.data.get("steps", ()),
                    )
                ),
                limit=3,
            )
        ),
    )
    if resume_attempt_dir is not None:
        source_dir = resume_attempt_dir
        source_policy_path = (
            source_dir / "policy.json"
            if (source_dir / "policy.json").exists()
            else source_dir / "team_plan.json"
        )
        source_report_path = next(
            path
            for path in (
                source_dir / "corrected_evaluation.json",
                source_dir / "evaluation.json",
                source_dir / "preliminary_physics_evaluation.json",
            )
            if path.exists()
        )
        source_policy = _load_policy(
            source_policy_path,
            allow_legacy_invalid_parameters=args.resume_failed_attempt_dir is not None,
        )
        if args.next_policy_attempt_index is not None:
            if args.next_policy_attempt_index <= source_policy.attempt:
                raise ValueError(
                    "--next-policy-attempt-index must be greater than the resumed "
                    f"policy attempt {source_policy.attempt}"
                )
            source_policy = replace(
                source_policy,
                attempt=args.next_policy_attempt_index - 1,
            )
        source_report = _load_evaluation_report(source_report_path)
        source_trace = _load_execution_trace(source_dir / "execution_trace.json")
        visual_payload = read_json(source_dir / "visual_evaluation.json")
        if visual_payload.get("summary"):
            source_report = replace(
                source_report,
                visual_summary=str(visual_payload["summary"]),
                evidence={
                    **source_report.evidence,
                    "visual_resume_evidence": visual_payload,
                },
            )
        if source_report.success and args.resume_failed_attempt_dir is not None:
            raise ValueError(
                "--resume-failed-attempt-dir requires a rejected rollout; use "
                f"--resume-attempt-dir for the accepted rollout at {source_dir}"
            )
        if source_report.evidence.get("resume_reclassification"):
            write_json(source_dir / "corrected_evaluation.json", source_report)
        if args.agent_mode == "model":
            def generate_resumed_policy(_generation_cycle: int):
                decision = evolver.decide_next_step(
                    source_policy,
                    source_report,
                    observation,
                    source_trace,
                    checkpoint_id=checkpoint_id,
                    # Persisted rollout evidence is not a new physical dispatch.
                    physically_dispatched=False,
                )
                next_policy, next_spec = evolver.evolve(
                    source_policy,
                    source_report,
                    observation,
                    source_trace,
                    controller_decision=decision,
                )
                return decision, next_policy, next_spec

            def record_resume_stall(stall_event: dict) -> None:
                cycle = int(stall_event["generation_cycle"])
                write_json(
                    args.output_dir
                    / f"resume_role_generation_stall_cycle_{cycle:02d}.json",
                    stall_event,
                )
                _publish_experiment_status(
                    role_pipeline,
                    "role_generation_stall",
                    attempt=source_policy.attempt,
                    checkpoint_id=checkpoint_id,
                    policy=source_policy,
                    report=source_report,
                    trace=source_trace,
                    rollback_performed=not source_report.success,
                    next_system_operation=(
                        "re-observe the accepted checkpoint and restart the complete "
                        "resumed Orchestrator/Mutation/Coder generation cycle"
                    ),
                    artifact_dir=args.output_dir,
                    details=stall_event,
                )

            def refresh_resume_generation(
                next_generation_cycle: int,
                stall_event: dict,
            ) -> None:
                nonlocal observation
                evolver.reset_generation_cycle(
                    source_policy.attempt,
                    stall_event=stall_event,
                )
                refreshed_state = _checkpoint_authoritative_planning_state(
                    live_runtime_state=read_json(args.runtime_state_file),
                    checkpoint_reference_state=checkpoint_reference_state,
                    turn_episode=turn_episode,
                )
                refreshed_visual = camera.observe(
                    label=(
                        "resume_generation_cycle_"
                        f"{next_generation_cycle:02d}"
                    ),
                    runtime_state=refreshed_state,
                )
                if args.visual_evaluator == "model":
                    _require_visual_frame(
                        refreshed_visual,
                        label="resumed generation recovery observation",
                    )
                observation = replace(
                    observation,
                    runtime_state=refreshed_state,
                    visual_observations=(refreshed_visual,),
                )
                print(
                    "[cap-box-carry] resumed ROLE_GENERATION_STALL; restarting "
                    f"complete generation cycle {next_generation_cycle}/"
                    f"{generation_stalls.maximum_consecutive_stalls} without "
                    "physical dispatch",
                    flush=True,
                )

            (
                resume_controller_decision,
                policy,
                resumed_spec,
            ) = _run_checkpoint_generation_cycles(
                tracker=generation_stalls,
                checkpoint_id=checkpoint_id,
                generate=generate_resumed_policy,
                on_stall=record_resume_stall,
                on_retry=refresh_resume_generation,
            )
        else:
            resume_controller_decision = evolution_controller.record_and_decide(
                policy=source_policy,
                report=source_report,
                trace=source_trace,
                checkpoint_id=checkpoint_id,
                physically_dispatched=False,
            )
            if resume_controller_decision.should_evolve:
                policy, resumed_spec = evolver.evolve(
                    source_policy,
                    source_report,
                    observation,
                )
            else:
                policy, resumed_spec = source_policy, ""
        stop_before_loop = not resume_controller_decision.should_evolve
        write_json(
            args.output_dir / "resume_evolution_controller_decision.json",
            resume_controller_decision.to_dict(),
        )
        if not source_report.success:
            # Resume consumes already-persisted evidence. Never count or promote
            # it as a new physical transaction in the destination run.
            resumed_record = EvolutionRecord(
                task=task,
                policy=source_policy,
                observation=observation,
                execution=source_trace,
                evaluation=source_report,
                evolved_spec=resumed_spec,
                next_script_id=policy.script_id,
                checkpoint_before=checkpoint_id,
                rollback_performed=True,
                controller_decision=resume_controller_decision.to_dict(),
            )
            records.append(resumed_record)
            write_json(source_dir / "policy.json", source_policy)
            write_json(source_dir / "resume_evaluation.json", source_report)
            _atomic_write_runner_json(
                source_dir / "evolution_record.json", resumed_record
            )
            append_rollout_evidence(
                args.output_dir / "raw_rollout_evidence.jsonl",
                resumed_spec,
                dry_run=not args.execute,
                policy=source_policy,
                report=source_report,
                trace=source_trace,
                checkpoint_before=checkpoint_id,
                controller_decision=resume_controller_decision.to_dict(),
                physically_dispatched=False,
                physical_transaction_completed=False,
            )
            if args.execute and args.agent_mode == "model":
                append_rollout_evidence(
                    args.raw_attempt_memory,
                    resumed_spec,
                    dry_run=False,
                    policy=source_policy,
                    report=source_report,
                    trace=source_trace,
                    checkpoint_before=checkpoint_id,
                    controller_decision=resume_controller_decision.to_dict(),
                    strategy_store=strategy_store,
                    strategy_phase=_strategy_phase(
                        task,
                        observation.runtime_state,
                        accepted_steps=episode_manifest.data.get("steps", ()),
                    ),
                    role_pipeline=role_pipeline,
                    physically_dispatched=False,
                    physical_transaction_completed=False,
                )
        write_json(
            args.output_dir / "resume_source.json",
            {
                "resume_attempt_dir": str(source_dir),
                "source_step_accepted": source_report.success,
                "source_script_id": source_policy.script_id,
                "next_script_id": policy.script_id,
                "next_attempt": policy.attempt,
                "checkpoint_id": checkpoint_id,
                "controller_action": resume_controller_decision.action,
                "controller_should_evolve": resume_controller_decision.should_evolve,
            },
        )
        source_outcome = "accepted" if source_report.success else "rejected"
        print(
            f"[cap-box-carry] evolved {source_outcome} rollout={source_policy.script_id} "
            f"directly into policy={policy.script_id} attempt={policy.attempt}",
            flush=True,
        )
    elif args.resume_policy is None:
        for generation_cycle in range(
            1,
            generation_stalls.maximum_consecutive_stalls + 1,
        ):
            try:
                policy = orchestrator.generate_initial_policy(
                    task,
                    observation,
                    attempt=args.next_policy_attempt_index or 1,
                )
                break
            except Exception as exc:
                role_stall = _role_stall_from_exception(exc)
                if role_stall is None:
                    raise
                stall_event = generation_stalls.record(
                    checkpoint_id=checkpoint_id,
                    generation_cycle=generation_cycle,
                    role=role_stall.role,
                    artifact_id=role_stall.artifact_id,
                    error=str(role_stall),
                    response_count=role_stall.response_count,
                    **_role_stall_record_fields(role_stall),
                )
                _publish_experiment_status(
                    role_pipeline,
                    "role_generation_stall",
                    attempt=args.next_policy_attempt_index or 1,
                    checkpoint_id=checkpoint_id,
                    next_system_operation=(
                        "re-observe the accepted checkpoint and restart the complete "
                        "Orchestrator-to-Coder generation cycle"
                    ),
                    artifact_dir=args.output_dir,
                    details=stall_event,
                )
                if stall_event["system_stall"]:
                    raise SystemStallError(
                        "three consecutive complete generation cycles stalled at "
                        f"checkpoint {checkpoint_id}"
                    ) from exc
                refreshed_state = _checkpoint_authoritative_planning_state(
                    live_runtime_state=read_json(args.runtime_state_file),
                    checkpoint_reference_state=checkpoint_reference_state,
                    turn_episode=turn_episode,
                )
                refreshed_visual = camera.observe(
                    label=f"initial_generation_cycle_{generation_cycle + 1:02d}",
                    runtime_state=refreshed_state,
                )
                if args.visual_evaluator == "model":
                    _require_visual_frame(
                        refreshed_visual,
                        label="initial generation recovery observation",
                    )
                observation = replace(
                    observation,
                    runtime_state=refreshed_state,
                    visual_observations=(refreshed_visual,),
                )
    else:
        policy = _load_policy(args.resume_policy)
        write_json(
            args.output_dir / "resume_source.json",
            {
                "resume_policy": str(args.resume_policy),
                "script_id": policy.script_id,
                "attempt": policy.attempt,
                "checkpoint_id": checkpoint_id,
            },
        )
        print(
            f"[cap-box-carry] resuming generated policy={policy.script_id} "
            f"attempt={policy.attempt} source={args.resume_policy}",
            flush=True,
        )
    _publish_experiment_status(
        role_pipeline,
        "policy_ready",
        attempt=policy.attempt,
        checkpoint_id=checkpoint_id,
        policy=policy,
        next_system_operation="capture pre-action state and run host preflight",
        artifact_dir=args.output_dir,
    )
    _print_team_assignment(policy)
    _persist_task_distribution(
        args.output_dir,
        policy,
        robot_ports={"front": backend.front_port, "back": backend.back_port},
    )
    print(
        f"[cap-box-carry] task distribution: {args.output_dir / 'latest_task_distribution.json'}",
        flush=True,
    )
    if args.max_attempts > 0 and policy.attempt > args.max_attempts:
        raise ValueError(
            f"resume policy attempt {policy.attempt} exceeds --max-attempts {args.max_attempts}"
        )
    attempt_numbers = (
        ()
        if stop_before_loop
        else (
            count(policy.attempt)
            if args.max_attempts == 0
            else range(policy.attempt, args.max_attempts + 1)
        )
    )
    for run_iteration, attempt in enumerate(attempt_numbers, start=1):
        codebase_guard.verify_unchanged(checkpoint=f"before_attempt_{attempt:02d}")
        if policy.attempt != attempt:
            raise RuntimeError(
                f"policy attempt sequence mismatch: loop={attempt}, policy={policy.attempt}"
            )
        attempt_dir = args.output_dir / f"attempt_{attempt:02d}"
        checkpoint_before = checkpoint_id
        # A generation failure can happen before decide_next_step() returns.
        # Keep this attempt from inheriting the previous attempt's decision.
        controller_decision = None
        transaction_commit_path: Path | None = None
        transaction_record_index: int | None = None
        pre_dispatch_recovery: dict = {}
        policy_action = _policy_action_method(policy)
        policy_phase_id = _policy_phase_id(policy)
        _publish_experiment_status(
            role_pipeline,
            "preparing_execution",
            attempt=attempt,
            checkpoint_id=checkpoint_id,
            policy=policy,
            next_system_operation=(
                "verify trusted support, capture the global pre-action frame, "
                "and run policy preflight"
            ),
            artifact_dir=attempt_dir,
        )
        if checkpoint_client is not None:
            print(
                "[cap-box-carry] restoring checkpoint="
                f"{checkpoint_id} before dispatch so observation and parent match",
                flush=True,
            )
            _publish_experiment_status(
                role_pipeline,
                "rolling_back",
                attempt=attempt,
                checkpoint_id=checkpoint_id,
                policy=policy,
                next_system_operation=(
                    "restore the trusted checkpoint before any new physical dispatch"
                ),
                artifact_dir=attempt_dir / "pre_dispatch_recovery",
                details={"reason": "pre_dispatch_parent_reassert"},
            )
            pre_dispatch_recovery = _restore_and_resynchronize(
                checkpoint_client=checkpoint_client,
                backend=backend,
                checkpoint_id=checkpoint_id,
                checkpoint_reference_state=checkpoint_reference_state,
                attempt_dir=attempt_dir / "pre_dispatch_recovery",
                heartbeat_file=args.heartbeat_file,
                runtime_state_file=args.runtime_state_file,
                idle_s=args.checkpoint_resync_idle_s,
                idle_keepalive=idle_keepalive,
                **ROLLBACK_POSE_LIMITS,
                safety_profile=args.physical_safety_profile,
                straight_heading_reference_rad=straight_heading_reference_rad,
                video_recorder=action_video_recorder,
                video_attempt=attempt,
                video_kind="checkpoint_restore",
                video_label=f"pre_dispatch_to_{checkpoint_id}",
                hold_planning=False,
            )
            write_json(attempt_dir / "pre_dispatch_recovery.json", pre_dispatch_recovery)
            _publish_experiment_status(
                role_pipeline,
                "rollback_complete",
                attempt=attempt,
                checkpoint_id=checkpoint_id,
                policy=policy,
                next_system_operation="resume host preflight from the trusted state",
                artifact_dir=attempt_dir / "pre_dispatch_recovery",
                details={"pre_dispatch_recovery": pre_dispatch_recovery},
            )
        if args.execute and policy_action in STRAIGHT_TRANSLATION_ACTIONS:
            live_heading = _straight_heading_status(
                read_json(args.runtime_state_file),
                reference_yaw_rad=straight_heading_reference_rad,
            )
            if not bool(live_heading.get("within_tolerance", False)):
                print(
                    "[cap-box-carry] restored carry heading drifted while planning; "
                    "normalizing yaw before the requested translation",
                    flush=True,
                )
                heading_normalization = _normalize_straight_heading_after_restore(
                    backend=backend,
                    heartbeat_file=args.heartbeat_file,
                    runtime_state_file=args.runtime_state_file,
                    output_dir=attempt_dir / "pre_dispatch_heading_normalization",
                    idle_keepalive=idle_keepalive,
                    reference_yaw_rad=straight_heading_reference_rad,
                    safety_profile=args.physical_safety_profile,
                )
                pre_dispatch_recovery = {
                    **pre_dispatch_recovery,
                    "heading_normalization": heading_normalization,
                }
                write_json(
                    attempt_dir / "pre_dispatch_recovery.json",
                    pre_dispatch_recovery,
                )
        api = G1SkillAPI(
            backend=backend,
            output_dir=attempt_dir / "skill_calls",
            heartbeat_file=args.heartbeat_file,
            runtime_state_file=args.runtime_state_file,
        )
        executor = PolicyExecutor(attempt_dir)
        robot_ports = {"front": backend.front_port, "back": backend.back_port}
        _persist_task_distribution(
            args.output_dir,
            policy,
            attempt_dir=attempt_dir,
            robot_ports=robot_ports,
        )
        pre_state = with_turn_episode(
            read_json(args.runtime_state_file),
            turn_episode,
        )
        pre_visual = camera.observe(
            label=f"attempt_{attempt:02d}_pre",
            runtime_state=pre_state,
        )
        if args.agent_mode == "model" and args.visual_evaluator == "model":
            _require_visual_frame(pre_visual, label=f"attempt {attempt} pre-dispatch")
        observation = ObservationBundle(
            runtime_state_path=str(args.runtime_state_file),
            runtime_state=pre_state,
            visual_observations=(pre_visual,),
            retrieved_specs=tuple(
                strategy_store.retrieve(
                    StrategyContext(
                        phase=_strategy_phase(
                            task,
                            pre_state,
                            accepted_steps=episode_manifest.data.get("steps", ()),
                        )
                    ),
                    limit=3,
                )
            ),
        )

        _persist_task_distribution(
            args.output_dir,
            policy,
            attempt_dir=attempt_dir,
            physical_dispatch_status="dispatching" if args.execute else "dry_run_not_dispatched",
            robot_ports=robot_ports,
        )
        _publish_experiment_status(
            role_pipeline,
            "preflight",
            attempt=attempt,
            checkpoint_id=checkpoint_id,
            policy=policy,
            next_system_operation=(
                "dispatch both robot scripts atomically"
                if args.execute
                else "validate the dry-run transaction without physical dispatch"
            ),
            artifact_dir=attempt_dir,
        )
        turn_episode_preflight_error = ""
        turn_episode_preflight_evidence: dict[str, object] = {}
        try:
            turn_episode_preflight_evidence = _validate_policy_turn_preflight(
                policy,
                runtime_state=pre_state,
                turn_episode=turn_episode,
                formal_v50_lite=bool(host_contract_protocol),
            )
        except TurnReadinessError as exc:
            turn_episode_preflight_error = str(exc)
            turn_episode_preflight_evidence = dict(exc.evidence)
        except TurnEpisodeError as exc:
            turn_episode_preflight_error = str(exc)
        if turn_episode_preflight_evidence:
            write_json(
                attempt_dir / "candidate_preflight.json",
                turn_episode_preflight_evidence,
            )
        if turn_episode_preflight_error:
            trace = ExecutionTrace(
                script_id=policy.script_id,
                attempt=policy.attempt,
                status="preflight_rejected",
                error=turn_episode_preflight_error,
                artifact_dir=str(attempt_dir),
                verification_contract=dict(policy.verification_contract),
            )
            print(
                "[cap-box-carry] turn-episode preflight rejected policy before physical "
                f"dispatch: {turn_episode_preflight_error}",
                flush=True,
            )
        elif args.stop_before_physical_dispatch:
            _persist_task_distribution(
                args.output_dir,
                policy,
                attempt_dir=attempt_dir,
                physical_dispatch_status="validated_not_dispatched",
                robot_ports=robot_ports,
            )
            write_json(
                args.output_dir / "preflight_only_result.json",
                {
                    "checkpoint_id": checkpoint_id,
                    "attempt": attempt,
                    "model_name": (
                        role_pipeline.model_name
                        if role_pipeline is not None
                        else ""
                    ),
                    "policy": asdict(policy),
                    "physical_dispatch": False,
                    "status": "ready_for_physical_dispatch",
                },
            )
            _publish_experiment_status(
                role_pipeline,
                "ready_for_physical_dispatch",
                attempt=attempt,
                checkpoint_id=checkpoint_id,
                policy=policy,
                next_system_operation=(
                    "operator or experiment runner may resume this trusted checkpoint "
                    "without --stop-before-physical-dispatch"
                ),
                artifact_dir=attempt_dir,
            )
            print(
                "[cap-box-carry] policy passed host preflight; stopped before "
                "physical dispatch by request",
                flush=True,
            )
            break
        else:
            _publish_experiment_status(
                role_pipeline,
                "executing" if args.execute else "dry_run",
                attempt=attempt,
                checkpoint_id=checkpoint_id,
                policy=policy,
                next_system_operation=(
                    "wait for both robot scripts, then evaluate physics and global view"
                ),
                artifact_dir=attempt_dir,
            )
            action_video_segment = (
                action_video_recorder.start_segment(
                    attempt=attempt,
                    kind="action",
                    label=policy_action or policy.script_id,
                    metadata={
                        "script_id": policy.script_id,
                        "checkpoint_before": checkpoint_id,
                    },
                )
                if args.execute
                else None
            )
            try:
                with _paused_idle_keepalive(idle_keepalive):
                    trace = executor.execute(policy, api)
                if args.execute and trace_has_physical_dispatch(trace):
                    # `PolicyExecutor` has returned its persisted execution trace.
                    # Mark the confirmed dispatch before camera/evaluator/checkpoint
                    # work can be interrupted; bind its outcome identity later.
                    transaction_commit_path = _begin_physical_transaction_commit(
                        attempt_dir=attempt_dir,
                        attempt=attempt,
                        script_id=policy.script_id,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after="",
                        checkpoint_committed=False,
                        memory_required=args.agent_mode == "model",
                        outcome_finalized=False,
                    )
            except BaseException as exc:
                action_video_recorder.stop_segment(
                    action_video_segment,
                    status="execution_exception",
                    tail_s=0.2,
                    metadata={
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            else:
                action_video_recorder.stop_segment(
                    action_video_segment,
                    status=str(trace.status),
                    tail_s=0.2,
                    metadata={"physically_dispatched": trace_has_physical_dispatch(trace)},
                )
        if args.execute and trace_has_physical_dispatch(trace):
            generation_stalls.reset_after_physical_transaction()
        _publish_experiment_status(
            role_pipeline,
            "evaluating",
            attempt=attempt,
            checkpoint_id=checkpoint_id,
            policy=policy,
            trace=trace,
            next_system_operation=(
                "combine physics state, contact evidence, and the global camera frame"
            ),
            artifact_dir=attempt_dir,
        )
        runtime_state = with_turn_episode(
            read_json(args.runtime_state_file),
            turn_episode,
        )
        post_action_stability: dict[str, object] = {}
        if args.execute and policy_action in SETTLED_CARRY_ACTIONS:
            post_action_stability = observe_payload_carry_stability(
                args.runtime_state_file,
                timeout_s=args.post_action_stability_timeout_s,
                stable_s=args.post_action_stable_s,
                safety_profile=args.physical_safety_profile,
            )
            runtime_state = read_json(args.runtime_state_file)
            write_json(
                attempt_dir / "post_action_stability_gate.json",
                post_action_stability,
            )
        post_visual = camera.observe(
            label=f"attempt_{attempt:02d}_post",
            runtime_state=runtime_state,
        )
        preliminary_report = (
            _turn_episode_preflight_report(
                turn_episode_preflight_error,
                policy_action=policy_action,
                turn_episode=turn_episode,
                preflight_evidence=turn_episode_preflight_evidence,
            )
            if turn_episode_preflight_error
            else evaluator.evaluate(
                task=task,
                trace=trace,
                visual_summary=post_visual.summary,
                before_state=pre_state,
                state_override=runtime_state,
                stepwise=True,
                straight_heading_reference_rad=straight_heading_reference_rad,
                turn_episode=turn_episode,
                route_phase_id=policy_phase_id,
                allow_pending_visual=True,
            )
        )
        if turn_episode_preflight_evidence and not turn_episode_preflight_error:
            preliminary_report = replace(
                preliminary_report,
                evidence={
                    **preliminary_report.evidence,
                    "candidate_preflight": turn_episode_preflight_evidence,
                },
            )
        preliminary_report = _attach_simulation_observation(
            preliminary_report,
            post_action_stability,
            evidence_key="post_action_stability_gate",
            failed_subtask=policy_action,
        )
        write_json(attempt_dir / "preliminary_physics_evaluation.json", preliminary_report)
        checkpoint_after = ""
        checkpoint_committed = False
        rollback_performed = False
        rollback_verification: dict = {}
        if (
            checkpoint_client is not None
            and not preliminary_report.success
            and trace_has_physical_dispatch(trace)
        ):
            _publish_experiment_status(
                role_pipeline,
                "rolling_back",
                attempt=attempt,
                checkpoint_id=checkpoint_id,
                policy=policy,
                report=preliminary_report,
                trace=trace,
                checkpoint_before=checkpoint_before,
                rollback_performed=True,
                next_system_operation=(
                    "restore the latest trusted checkpoint before model failure analysis"
                ),
                artifact_dir=attempt_dir,
            )
            rollback_verification = _restore_and_resynchronize(
                checkpoint_client=checkpoint_client,
                backend=backend,
                checkpoint_id=checkpoint_id,
                checkpoint_reference_state=checkpoint_reference_state,
                attempt_dir=attempt_dir,
                heartbeat_file=args.heartbeat_file,
                runtime_state_file=args.runtime_state_file,
                idle_s=args.checkpoint_resync_idle_s,
                idle_keepalive=idle_keepalive,
                **ROLLBACK_POSE_LIMITS,
                safety_profile=args.physical_safety_profile,
                straight_heading_reference_rad=straight_heading_reference_rad,
                video_recorder=action_video_recorder,
                video_attempt=attempt,
                video_kind="rollback",
                video_label=f"physical_guard_to_{checkpoint_id}",
                hold_planning=True,
            )
            rollback_performed = True
            print(
                f"[cap-box-carry] physical guard rejected step; restored checkpoint={checkpoint_id} "
                f"before model evaluation (failure={preliminary_report.failure_type})",
                flush=True,
            )
            _publish_experiment_status(
                role_pipeline,
                "rollback_complete",
                attempt=attempt,
                checkpoint_id=checkpoint_id,
                policy=policy,
                report=preliminary_report,
                trace=trace,
                checkpoint_before=checkpoint_before,
                checkpoint_after=checkpoint_id,
                rollback_performed=True,
                next_system_operation=(
                    "run VLM failure analysis on the failed endpoint and trusted rollback view"
                ),
                artifact_dir=attempt_dir,
                details={"rollback_verification": rollback_verification},
            )
        visual_report = None
        try:
            if visual_verifier is not None:
                for evaluation_cycle in range(
                    1,
                    generation_stalls.maximum_consecutive_stalls + 1,
                ):
                    try:
                        visual_report = visual_verifier.evaluate(
                            task=task,
                            trace=trace,
                            visual=post_visual,
                            runtime_state=runtime_state,
                            before_state=pre_state,
                            physics_report=preliminary_report,
                        )
                        generation_stalls.reset_after_successful_role_cycle()
                        break
                    except Exception as exc:
                        role_stall = _role_stall_from_exception(exc)
                        if role_stall is None:
                            raise
                        stall_event = generation_stalls.record(
                            checkpoint_id=checkpoint_id,
                            generation_cycle=evaluation_cycle,
                            role=role_stall.role,
                            artifact_id=role_stall.artifact_id,
                            error=str(role_stall),
                            response_count=role_stall.response_count,
                            **_role_stall_record_fields(role_stall),
                        )
                        stall_event["pipeline_stage"] = "post_execution_evaluation"
                        write_json(
                            attempt_dir
                            / f"evaluator_role_stall_cycle_{evaluation_cycle:02d}.json",
                            stall_event,
                        )
                        if stall_event["system_stall"]:
                            raise SystemStallError(
                                "Evaluator/Failure Analyzer stalled for three complete "
                                f"cycles at checkpoint {checkpoint_id}"
                            ) from exc
                write_json(attempt_dir / "visual_evaluation.json", visual_report)
                post_visual = replace(post_visual, summary=visual_report.summary)
        except BaseException as evaluation_exc:
            if args.execute and trace_has_physical_dispatch(trace):
                if checkpoint_client is not None and not rollback_performed:
                    rollback_verification = _restore_and_resynchronize(
                        checkpoint_client=checkpoint_client,
                        backend=backend,
                        checkpoint_id=checkpoint_id,
                        checkpoint_reference_state=checkpoint_reference_state,
                        attempt_dir=attempt_dir / "post_execution_system_rollback",
                        heartbeat_file=args.heartbeat_file,
                        runtime_state_file=args.runtime_state_file,
                        idle_s=args.checkpoint_resync_idle_s,
                        idle_keepalive=idle_keepalive,
                        **ROLLBACK_POSE_LIMITS,
                        safety_profile=args.physical_safety_profile,
                        straight_heading_reference_rad=straight_heading_reference_rad,
                        video_recorder=action_video_recorder,
                        video_attempt=attempt,
                        video_kind="rollback",
                        video_label=f"system_error_to_{checkpoint_id}",
                        hold_planning=True,
                    )
                    rollback_performed = True
                terminal_report = _post_dispatch_model_terminal_report(
                    preliminary_report,
                    error=evaluation_exc,
                    failed_subtask=policy_action,
                    rollback_verification=rollback_verification,
                )
                terminal_report = _with_checkpoint_outcome(
                    terminal_report,
                    route_discovery=bool(task.route_discovery),
                    checkpoint_committed=False,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_id,
                    rollback_performed=rollback_performed,
                    physically_dispatched=True,
                )
                if transaction_commit_path is None:
                    transaction_commit_path = _begin_physical_transaction_commit(
                        attempt_dir=attempt_dir,
                        attempt=attempt,
                        script_id=policy.script_id,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after="",
                        checkpoint_committed=False,
                        memory_required=args.agent_mode == "model",
                        outcome_finalized=False,
                    )
                _finalize_physical_transaction_outcome(
                    transaction_commit_path,
                    checkpoint_after=checkpoint_id,
                    checkpoint_committed=False,
                    step_accepted=False,
                )
                terminal_observation = replace(
                    observation,
                    runtime_state=read_json(args.runtime_state_file),
                    visual_observations=(pre_visual, post_visual),
                )
                terminal_decision = evolution_controller.record_and_decide(
                    policy=policy,
                    report=terminal_report,
                    trace=trace,
                    checkpoint_id=checkpoint_before,
                    physically_dispatched=True,
                )
                _advance_physical_transaction_commit(
                    transaction_commit_path,
                    "controller_accounting",
                    details={
                        "physical_iterations": terminal_decision.physical_iterations,
                        "controller_action": terminal_decision.action,
                    },
                )
                terminal_record = EvolutionRecord(
                    task=task,
                    policy=policy,
                    observation=terminal_observation,
                    execution=trace,
                    evaluation=terminal_report,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_id,
                    rollback_performed=rollback_performed,
                    controller_decision=terminal_decision.to_dict(),
                )
                records.append(terminal_record)
                write_json(attempt_dir / "evaluation.json", terminal_report)
                write_json(
                    attempt_dir / "evolution_controller_decision.json",
                    terminal_decision.to_dict(),
                )
                _advance_physical_transaction_commit(
                    transaction_commit_path,
                    "attempt_artifacts",
                    details={
                        "evaluation": "evaluation.json",
                        "controller_decision": "evolution_controller_decision.json",
                    },
                )
                _atomic_write_runner_json(
                    attempt_dir / "evolution_record.json", terminal_record
                )
                _atomic_write_runner_jsonl(
                    args.output_dir / "evolution_records.jsonl", records
                )
                _advance_physical_transaction_commit(
                    transaction_commit_path,
                    "evolution_record",
                    details={
                        "path": "evolution_record.json",
                        "state": "post_execution_system_terminal",
                    },
                )
                terminal_raw_rollout = append_rollout_evidence(
                    args.output_dir / "raw_rollout_evidence.jsonl",
                    "",
                    dry_run=False,
                    policy=policy,
                    report=terminal_report,
                    trace=trace,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_id,
                    controller_decision=terminal_decision.to_dict(),
                    strategy_phase=_strategy_phase(
                        task,
                        terminal_observation.runtime_state,
                        accepted_steps=episode_manifest.data.get("steps", ()),
                    ),
                    physically_dispatched=True,
                    physical_transaction_completed=False,
                )
                _advance_physical_transaction_commit(
                    transaction_commit_path,
                    "raw_rollout",
                    details={
                        "path": "raw_rollout_evidence.jsonl",
                        "rollout_id": str(terminal_raw_rollout.get("rollout_id", "")),
                    },
                )
                if args.agent_mode == "model":
                    terminal_memory_rollout = append_rollout_evidence(
                        args.raw_attempt_memory,
                        "",
                        dry_run=False,
                        policy=policy,
                        report=terminal_report,
                        trace=trace,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=checkpoint_id,
                        controller_decision=terminal_decision.to_dict(),
                        strategy_store=strategy_store,
                        strategy_phase=_strategy_phase(
                            task,
                            terminal_observation.runtime_state,
                            accepted_steps=episode_manifest.data.get("steps", ()),
                        ),
                        role_pipeline=None,
                        physically_dispatched=True,
                        physical_transaction_completed=False,
                    )
                    _advance_physical_transaction_commit(
                        transaction_commit_path,
                        "memory_update",
                        details={
                            "path": str(args.raw_attempt_memory),
                            "rollout_id": str(
                                terminal_memory_rollout.get("rollout_id", "")
                            ),
                            "status": str(
                                terminal_memory_rollout.get("memory_update", {}).get(
                                    "status", ""
                                )
                                if isinstance(
                                    terminal_memory_rollout.get("memory_update"), dict
                                )
                                else ""
                            ),
                        },
                    )
                _persist_task_distribution(
                    args.output_dir,
                    policy,
                    attempt_dir=attempt_dir,
                    physical_dispatch_status="post_execution_system_failure_restored",
                    robot_ports=robot_ports,
                    trace=trace,
                    report=terminal_report,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_id,
                    rollback_performed=rollback_performed,
                )
            raise
        precommit_stability: dict[str, object] = {}
        if (
            checkpoint_client is not None
            and not rollback_performed
            and args.execute
            and policy_action in SETTLED_CARRY_ACTIONS
        ):
            precommit_stability = observe_payload_carry_stability(
                args.runtime_state_file,
                timeout_s=args.precommit_stability_timeout_s,
                stable_s=args.precommit_stable_s,
                safety_profile=args.physical_safety_profile,
            )
            write_json(
                attempt_dir / "precommit_stability_gate.json",
                precommit_stability,
            )
        if checkpoint_client is not None and not rollback_performed:
            runtime_state = read_json(args.runtime_state_file)
        report = (
            _turn_episode_preflight_report(
                turn_episode_preflight_error,
                policy_action=policy_action,
                turn_episode=turn_episode,
                visual_report=visual_report,
                preflight_evidence=turn_episode_preflight_evidence,
            )
            if turn_episode_preflight_error
            else evaluator.evaluate(
                task=task,
                trace=trace,
                visual_summary=post_visual.summary,
                visual_report=visual_report,
                before_state=pre_state,
                state_override=runtime_state,
                stepwise=True,
                straight_heading_reference_rad=straight_heading_reference_rad,
                turn_episode=turn_episode,
                route_phase_id=policy_phase_id,
            )
        )
        report = _attach_simulation_observation(
            report,
            post_action_stability,
            evidence_key="post_action_stability_gate",
            failed_subtask=policy_action,
        )
        report = _attach_simulation_observation(
            report,
            precommit_stability,
            evidence_key="precommit_stability_gate",
            failed_subtask=policy_action,
        )
        if rollback_verification:
            report = replace(
                report,
                evidence={**report.evidence, "rollback_verification": rollback_verification},
            )
        if pre_dispatch_recovery:
            report = replace(
                report,
                evidence={**report.evidence, "pre_dispatch_recovery": pre_dispatch_recovery},
            )
        transition_decision = verify_transition(
            task=task,
            report=report,
            trace=trace,
            before_state=pre_state,
            after_state=runtime_state,
            accepted_manifest=episode_manifest.data,
        )
        report = apply_transition_decision(report, transition_decision)
        write_json(
            attempt_dir / "transition_decision.json",
            transition_decision.to_dict(),
        )
        evolution_runtime_state = _checkpoint_authoritative_planning_state(
            live_runtime_state=(
                read_json(args.runtime_state_file) if rollback_performed else runtime_state
            ),
            checkpoint_reference_state=checkpoint_reference_state,
            turn_episode=turn_episode if isinstance(turn_episode, dict) else {},
        )
        observation = ObservationBundle(
            runtime_state_path=str(args.runtime_state_file),
            runtime_state=evolution_runtime_state,
            visual_observations=(pre_visual, post_visual),
            retrieved_specs=tuple(
                strategy_store.retrieve(
                    StrategyContext(
                        phase=_strategy_phase(
                            task,
                            evolution_runtime_state,
                            accepted_steps=episode_manifest.data.get("steps", ()),
                        ),
                        last_failure_type=("" if report.success else report.failure_type),
                        last_failure_level=(
                            "" if report.success else classify_report(report).level
                        ),
                    ),
                    limit=3,
                )
            ),
        )
        if checkpoint_client is not None:
            if report.success and policy_action in CHECKPOINT_ADVANCING_ACTIONS:
                _publish_experiment_status(
                    role_pipeline,
                    "checkpoint_validation",
                    attempt=attempt,
                    checkpoint_id=checkpoint_id,
                    policy=policy,
                    report=report,
                    trace=trace,
                    checkpoint_before=checkpoint_before,
                    next_system_operation=(
                        "save and round-trip the candidate checkpoint before accepting the step"
                    ),
                    artifact_dir=attempt_dir / "checkpoint_candidate_roundtrip",
                )
                candidate_step = accepted_steps + 1
                candidate_checkpoint_id = f"accepted_step_{candidate_step:02d}"
                with _checkpoint_heartbeat_lease(idle_keepalive):
                    saved_checkpoint = checkpoint_client.save(candidate_checkpoint_id)
                candidate_reference_state = saved_checkpoint.runtime_state_after or read_json(
                    args.runtime_state_file
                )
                candidate_heading_reference_rad = (
                    _straight_heading_reference_after_accepted_action(
                        current_reference_rad=straight_heading_reference_rad,
                        trace=trace,
                        runtime_state=candidate_reference_state,
                    )
                )
                roundtrip_dir = attempt_dir / "checkpoint_candidate_roundtrip"
                roundtrip_restore_verification: dict[str, object] = {}
                try:
                    roundtrip_restore_verification = _restore_and_resynchronize(
                        checkpoint_client=checkpoint_client,
                        backend=backend,
                        checkpoint_id=candidate_checkpoint_id,
                        checkpoint_reference_state=candidate_reference_state,
                        attempt_dir=roundtrip_dir,
                        heartbeat_file=args.heartbeat_file,
                        runtime_state_file=args.runtime_state_file,
                        idle_s=args.checkpoint_resync_idle_s,
                        idle_keepalive=idle_keepalive,
                        **ROLLBACK_POSE_LIMITS,
                        safety_profile=args.physical_safety_profile,
                        straight_heading_reference_rad=(
                            candidate_heading_reference_rad
                        ),
                        candidate_checkpoint=True,
                    )
                    checkpoint_stable_s = _checkpoint_roundtrip_stable_s(
                        policy_action
                    )
                    checkpoint_roundtrip_stability = (
                        _observe_checkpoint_roundtrip_stability(
                            runtime_state_file=args.runtime_state_file,
                            policy_action=policy_action,
                            stable_s=checkpoint_stable_s,
                            safety_profile=args.physical_safety_profile,
                        )
                    )
                    roundtrip_semantic_report = evaluator.evaluate(
                        task=task,
                        trace=trace,
                        visual_summary=(
                            "Deterministic checkpoint round-trip semantic verification."
                        ),
                        before_state=pre_state,
                        state_override=read_json(args.runtime_state_file),
                        stepwise=True,
                        straight_heading_reference_rad=(
                            candidate_heading_reference_rad
                        ),
                        turn_episode=turn_episode,
                        route_phase_id=policy_phase_id,
                        # The fresh global frame was already evaluated before the
                        # candidate was saved. This second pass is a deterministic
                        # physics check after planner-reset restore, so requiring a
                        # duplicate image would misclassify a valid semantic result
                        # as missing evidence.
                        allow_pending_visual=True,
                    )
                except Exception as exc:
                    system_invalid = isinstance(
                        exc,
                        (
                            CheckpointBackendFatalError,
                            TimeoutError,
                            ConnectionError,
                            OSError,
                        ),
                    )
                    checkpoint_roundtrip_stability = {
                        "success": False,
                        "failure_type": (
                            "runtime_state_missing"
                            if system_invalid
                            else "payload_unstable"
                        ),
                        "reason": (
                            "candidate checkpoint could not survive planner reset and "
                            f"physical round-trip: {type(exc).__name__}: {exc}"
                        ),
                        "error_type": type(exc).__name__,
                        "failure_owner": (
                            "system" if system_invalid else "physical_endpoint"
                        ),
                    }
                    roundtrip_semantic_report = None
                    if system_invalid:
                        raise
                write_json(
                    attempt_dir / "checkpoint_roundtrip_stability_gate.json",
                    checkpoint_roundtrip_stability,
                )
                if roundtrip_semantic_report is not None:
                    write_json(
                        attempt_dir
                        / "checkpoint_roundtrip_semantic_evaluation.json",
                        asdict(roundtrip_semantic_report),
                    )
                if roundtrip_restore_verification:
                    report = replace(
                        report,
                        evidence={
                            **report.evidence,
                            "checkpoint_roundtrip_restore_verification": (
                                roundtrip_restore_verification
                            ),
                        },
                    )
                report = _attach_simulation_observation(
                    report,
                    checkpoint_roundtrip_stability,
                    evidence_key="checkpoint_roundtrip_stability_gate",
                    failed_subtask=policy_action,
                )
                if roundtrip_semantic_report is not None:
                    report = _attach_checkpoint_roundtrip_semantics(
                        report,
                        roundtrip_semantic_report,
                    )
                if report.success:
                    accepted_steps = candidate_step
                    checkpoint_after = candidate_checkpoint_id
                    checkpoint_reference_state = candidate_reference_state
                    checkpoint_id = candidate_checkpoint_id
                    straight_heading_reference_rad = candidate_heading_reference_rad
                    straight_heading_references[checkpoint_id] = (
                        straight_heading_reference_rad
                    )
                    _write_straight_heading_references(
                        straight_heading_ledger_path,
                        active_checkpoint_id=checkpoint_id,
                        references=straight_heading_references,
                    )
                    checkpoint_committed = True
                    print(
                        "[cap-box-carry] step accepted after checkpoint round-trip; "
                        f"committed checkpoint={checkpoint_after}",
                        flush=True,
                    )
                else:
                    _publish_experiment_status(
                        role_pipeline,
                        "rolling_back",
                        attempt=attempt,
                        checkpoint_id=checkpoint_id,
                        policy=policy,
                        report=report,
                        trace=trace,
                        checkpoint_before=checkpoint_before,
                        rollback_performed=True,
                        next_system_operation=(
                            "discard the unstable candidate and restore its trusted parent"
                        ),
                        artifact_dir=(
                            attempt_dir / "checkpoint_candidate_rejected_restore"
                        ),
                    )
                    rollback_verification = _restore_and_resynchronize(
                        checkpoint_client=checkpoint_client,
                        backend=backend,
                        checkpoint_id=checkpoint_id,
                        checkpoint_reference_state=checkpoint_reference_state,
                        attempt_dir=attempt_dir / "checkpoint_candidate_rejected_restore",
                        heartbeat_file=args.heartbeat_file,
                        runtime_state_file=args.runtime_state_file,
                        idle_s=args.checkpoint_resync_idle_s,
                        idle_keepalive=idle_keepalive,
                        **ROLLBACK_POSE_LIMITS,
                        safety_profile=args.physical_safety_profile,
                        straight_heading_reference_rad=straight_heading_reference_rad,
                        video_recorder=action_video_recorder,
                        video_attempt=attempt,
                        video_kind="rollback",
                        video_label=f"candidate_rejected_to_{checkpoint_id}",
                        hold_planning=True,
                    )
                    rollback_performed = True
                    report = replace(
                        report,
                        evidence={
                            **report.evidence,
                            "rollback_verification": rollback_verification,
                        },
                    )
                    print(
                        "[cap-box-carry] candidate checkpoint failed round-trip stability; "
                        f"restored parent checkpoint={checkpoint_id}",
                        flush=True,
                    )
                    _publish_experiment_status(
                        role_pipeline,
                        "rollback_complete",
                        attempt=attempt,
                        checkpoint_id=checkpoint_id,
                        policy=policy,
                        report=report,
                        trace=trace,
                        checkpoint_before=checkpoint_before,
                        checkpoint_after=checkpoint_id,
                        rollback_performed=True,
                        next_system_operation=(
                            "analyze why the candidate failed checkpoint stability"
                        ),
                        artifact_dir=(
                            attempt_dir / "checkpoint_candidate_rejected_restore"
                        ),
                        details={"rollback_verification": rollback_verification},
                    )
            elif report.success:
                checkpoint_after = checkpoint_id
                print(
                    "[cap-box-carry] safe non-advancing action accepted; retained "
                    f"productive checkpoint={checkpoint_id}",
                    flush=True,
                )
            elif not rollback_performed and trace_has_physical_dispatch(trace):
                _publish_experiment_status(
                    role_pipeline,
                    "rolling_back",
                    attempt=attempt,
                    checkpoint_id=checkpoint_id,
                    policy=policy,
                    report=report,
                    trace=trace,
                    checkpoint_before=checkpoint_before,
                    rollback_performed=True,
                    next_system_operation="restore the latest trusted checkpoint",
                    artifact_dir=attempt_dir,
                )
                rollback_verification = _restore_and_resynchronize(
                    checkpoint_client=checkpoint_client,
                    backend=backend,
                    checkpoint_id=checkpoint_id,
                    checkpoint_reference_state=checkpoint_reference_state,
                    attempt_dir=attempt_dir,
                    heartbeat_file=args.heartbeat_file,
                    runtime_state_file=args.runtime_state_file,
                    idle_s=args.checkpoint_resync_idle_s,
                    idle_keepalive=idle_keepalive,
                    **ROLLBACK_POSE_LIMITS,
                    safety_profile=args.physical_safety_profile,
                    straight_heading_reference_rad=straight_heading_reference_rad,
                    video_recorder=action_video_recorder,
                    video_attempt=attempt,
                    video_kind="rollback",
                    video_label=f"verifier_rejected_to_{checkpoint_id}",
                    hold_planning=True,
                )
                rollback_performed = True
                report = replace(
                    report,
                    evidence={**report.evidence, "rollback_verification": rollback_verification},
                )
                print(
                    f"[cap-box-carry] step rejected; restored checkpoint={checkpoint_id}",
                    flush=True,
                )
                _publish_experiment_status(
                    role_pipeline,
                    "rollback_complete",
                    attempt=attempt,
                    checkpoint_id=checkpoint_id,
                    policy=policy,
                    report=report,
                    trace=trace,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_id,
                    rollback_performed=True,
                    next_system_operation="analyze the rejected rollout and repair its strategy",
                    artifact_dir=attempt_dir,
                        details={"rollback_verification": rollback_verification},
                    )
        physically_dispatched = bool(
            args.execute and trace_has_physical_dispatch(trace)
        )
        effective_checkpoint_after = str(checkpoint_after or checkpoint_id)
        report = _with_checkpoint_outcome(
            report,
            route_discovery=bool(task.route_discovery),
            checkpoint_committed=checkpoint_committed,
            checkpoint_before=checkpoint_before,
            checkpoint_after=effective_checkpoint_after,
            rollback_performed=rollback_performed,
            physically_dispatched=physically_dispatched,
        )
        if physically_dispatched:
            if transaction_commit_path is None:
                transaction_commit_path = _begin_physical_transaction_commit(
                    attempt_dir=attempt_dir,
                    attempt=attempt,
                    script_id=policy.script_id,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after="",
                    checkpoint_committed=False,
                    memory_required=args.agent_mode == "model",
                    outcome_finalized=False,
                )
            _finalize_physical_transaction_outcome(
                transaction_commit_path,
                checkpoint_after=effective_checkpoint_after,
                checkpoint_committed=checkpoint_committed,
                step_accepted=bool(report.success),
            )
            # Commit the physical budget/accounting row before Memory Update or
            # any next-policy model role. The later decision call idempotently
            # enriches this same row with model-owned repair evidence.
            accounting_decision = evolution_controller.record_and_decide(
                policy=policy,
                report=report,
                trace=trace,
                checkpoint_id=checkpoint_before,
                physically_dispatched=True,
                hard_iteration_limit_reached=(
                    args.max_attempts > 0 and attempt >= args.max_attempts
                ),
            )
            _advance_physical_transaction_commit(
                transaction_commit_path,
                "controller_accounting",
                details={
                    "physical_iterations": accounting_decision.physical_iterations,
                    "controller_action": accounting_decision.action,
                },
            )
        if host_contract_protocol:
            _record_formal_shadow_physical_result(
                args.output_dir,
                attempt=attempt,
                report=report,
                trace=trace,
            )
        if report.success and trace_has_physical_dispatch(trace):
            _archive_accepted_action_visuals(
                output_dir=args.output_dir,
                attempt=attempt,
                checkpoint_id=effective_checkpoint_after,
                policy=policy,
                post_visual=post_visual,
                report=report,
            )
        turn_episode = update_turn_episode(
            turn_episode,
            policy=policy,
            report=report,
            visual_report=visual_report,
            checkpoint_id=effective_checkpoint_after,
        )
        if checkpoint_committed:
            turn_episode = observe_passage_alignment(
                turn_episode,
                visual_report=visual_report,
                checkpoint_id=effective_checkpoint_after,
            )
        write_json(turn_episode_path, turn_episode)
        evolution_runtime_state = with_turn_episode(
            evolution_runtime_state,
            turn_episode,
        )
        observation = replace(
            observation,
            runtime_state=evolution_runtime_state,
        )
        if checkpoint_committed:
            # The next model decision starts from the newly committed state.  Keeping the
            # pre-save checkpoint tag here incorrectly carries rejected transactions from the
            # parent state into the new branch and can exhaust an otherwise valid candidate.
            if checkpoint_client is not None:
                _restore_and_resynchronize(
                    checkpoint_client=checkpoint_client,
                    backend=backend,
                    checkpoint_id=checkpoint_id,
                    checkpoint_reference_state=checkpoint_reference_state,
                    attempt_dir=attempt_dir / "post_accept_planning_hold",
                    heartbeat_file=args.heartbeat_file,
                    runtime_state_file=args.runtime_state_file,
                    idle_s=args.checkpoint_resync_idle_s,
                    idle_keepalive=idle_keepalive,
                    **ROLLBACK_POSE_LIMITS,
                    safety_profile=args.physical_safety_profile,
                    straight_heading_reference_rad=straight_heading_reference_rad,
                    hold_planning=True,
                )
            evolution_runtime_state = with_turn_episode(
                _checkpoint_authoritative_planning_state(
                    live_runtime_state=read_json(args.runtime_state_file),
                    checkpoint_reference_state=checkpoint_reference_state,
                    turn_episode=turn_episode,
                ),
                turn_episode,
            )
            accepted_route_quality = episode_manifest.append_accepted(
                policy=policy,
                trace=trace,
                report=report,
                checkpoint_before=checkpoint_before,
                checkpoint_after=effective_checkpoint_after,
                source_artifact_dir=attempt_dir,
            )
            if accepted_route_quality:
                report = replace(
                    report,
                    evidence={
                        **report.evidence,
                        "accepted_route_quality": accepted_route_quality,
                    },
                    recommendations=tuple(report.recommendations)
                    + (
                        "Keep this physically safe checkpoint, but do not promote "
                        "the action as positive route memory until later motion "
                        "establishes connected route progress.",
                    ),
                )
            exemplar_recorded = False
            if args.agent_mode == "model":
                exemplar_recorded = action_memory.record_accepted(
                    policy=policy,
                    trace=trace,
                    report=report,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=effective_checkpoint_after,
                    strategy_phase=_strategy_phase(
                        task,
                        evolution_runtime_state,
                        accepted_steps=episode_manifest.data.get("steps", ()),
                    ),
                    source_artifact_dir=attempt_dir,
                )
            _advance_physical_transaction_commit(
                transaction_commit_path,
                "checkpoint_lineage",
                details={
                    "manifest": str(episode_manifest.path),
                    "latest_checkpoint": effective_checkpoint_after,
                    "action_exemplar_recorded": exemplar_recorded,
                },
            )
            print(
                "[cap-box-carry] accepted episode updated "
                f"checkpoint={effective_checkpoint_after} "
                f"action_exemplar_recorded={exemplar_recorded}",
                flush=True,
            )
            observation = replace(
                observation,
                runtime_state=evolution_runtime_state,
                retrieved_specs=tuple(
                    strategy_store.retrieve(
                        StrategyContext(
                            phase=_strategy_phase(
                                task,
                                evolution_runtime_state,
                                accepted_steps=episode_manifest.data.get("steps", ()),
                            ),
                        ),
                        limit=3,
                    )
                ),
            )
        elif rollback_performed:
            evolution_runtime_state = with_turn_episode(
                read_json(args.runtime_state_file),
                turn_episode,
            )
            rollback_visual = camera.observe(
                label=f"attempt_{attempt:02d}_rollback_current",
                runtime_state=evolution_runtime_state,
            )
            write_json(
                attempt_dir / "trusted_checkpoint_observation.json",
                {
                    "frame_path": rollback_visual.frame_path,
                    "checkpoint_id": checkpoint_id,
                    "model_role_invoked": False,
                    "consumer": "next_task_orchestrator_or_mutation_cycle",
                },
            )
            failed_endpoint_visual, rollback_visual = _label_rollback_visual_timeline(
                post_visual,
                rollback_visual,
            )
            observation = replace(
                observation,
                runtime_state=evolution_runtime_state,
                visual_observations=(failed_endpoint_visual, rollback_visual),
                retrieved_specs=tuple(
                    strategy_store.retrieve(
                        StrategyContext(
                            phase=_strategy_phase(
                                task,
                                evolution_runtime_state,
                                accepted_steps=episode_manifest.data.get("steps", ()),
                            ),
                            last_failure_type=report.failure_type,
                            last_failure_level=classify_report(report).level,
                        ),
                        limit=3,
                    )
                ),
            )
        else:
            observation = replace(
                observation,
                runtime_state=evolution_runtime_state,
            )
        if args.execute:
            dispatch_status = (
                "accepted_checkpoint_committed"
                if checkpoint_committed
                else "accepted_checkpoint_retained"
                if report.success
                else "rejected_before_dispatch"
                if not trace_has_physical_dispatch(trace)
                else "rejected_checkpoint_restored"
            )
        else:
            dispatch_status = "dry_run_completed"
        _persist_task_distribution(
            args.output_dir,
            policy,
            attempt_dir=attempt_dir,
            physical_dispatch_status=dispatch_status,
            robot_ports=robot_ports,
            trace=trace,
            report=report,
            checkpoint_before=checkpoint_before,
            checkpoint_after=effective_checkpoint_after,
            rollback_performed=rollback_performed,
        )
        write_json(attempt_dir / "policy.json", policy)
        write_json(attempt_dir / "evaluation.json", report)
        _advance_physical_transaction_commit(
            transaction_commit_path,
            "attempt_artifacts",
            details={"policy": "policy.json", "evaluation": "evaluation.json"},
        )
        task_complete = bool(report.metrics.get("task_complete", 0.0))
        final_runtime_status = (
            "dry_run_complete"
            if not args.execute
            else "task_complete"
            if report.success and task_complete
            else "accepted"
            if report.success
            else "rejected"
        )
        _publish_experiment_status(
            role_pipeline,
            final_runtime_status,
            attempt=attempt,
            checkpoint_id=effective_checkpoint_after,
            policy=policy,
            report=report,
            trace=trace,
            checkpoint_before=checkpoint_before,
            checkpoint_after=effective_checkpoint_after,
            rollback_performed=rollback_performed,
            next_system_operation=(
                "stop the evolution loop; the physics verifier confirmed task completion"
                if final_runtime_status == "task_complete"
                else "update the global phase and plan the next subtask"
                if report.success
                else "run Failure Analyzer and Evolution Manager, then repair the subtask"
            ),
            artifact_dir=attempt_dir,
        )
        if physically_dispatched:
            generation_stalls.reset_after_physical_transaction()
        physical_transaction_completed = bool(
            physically_dispatched
            and not (
                isinstance(
                    report.evidence.get("post_execution_model_terminal", {}),
                    dict,
                )
                and report.evidence.get("post_execution_model_terminal", {})
            )
            and (
                trace_physical_transaction_completed(trace)
                or _monitored_physical_failure(trace) is not None
            )
        )

        # Finalize this transaction before any model is asked to create the
        # next one. Raw evidence is durable first; Memory Update is best effort.
        raw_rollout = append_rollout_evidence(
            args.output_dir / "raw_rollout_evidence.jsonl",
            "",
            dry_run=not args.execute,
            policy=policy,
            report=report,
            trace=trace,
            checkpoint_before=checkpoint_before,
            checkpoint_after=effective_checkpoint_after,
            strategy_phase=_strategy_phase(
                task,
                evolution_runtime_state,
                accepted_steps=episode_manifest.data.get("steps", ()),
            ),
            physically_dispatched=physically_dispatched,
            physical_transaction_completed=physical_transaction_completed,
        )
        _advance_physical_transaction_commit(
            transaction_commit_path,
            "raw_rollout",
            details={
                "path": "raw_rollout_evidence.jsonl",
                "rollout_id": str(raw_rollout.get("rollout_id", "")),
            },
        )
        if physically_dispatched:
            pending_record = EvolutionRecord(
                task=task,
                policy=policy,
                observation=observation,
                execution=trace,
                evaluation=report,
                checkpoint_before=checkpoint_before,
                checkpoint_after=effective_checkpoint_after,
                rollback_performed=rollback_performed,
                controller_decision={
                    "status": "physical_transaction_finalized_pending_generation",
                    "generation_interrupted": False,
                    "next_policy_generated": False,
                },
            )
            transaction_record_index = len(records)
            records.append(pending_record)
            _atomic_write_runner_json(
                attempt_dir / "evolution_record.json", pending_record
            )
            _atomic_write_runner_jsonl(
                args.output_dir / "evolution_records.jsonl", records
            )
            _advance_physical_transaction_commit(
                transaction_commit_path,
                "evolution_record",
                details={
                    "path": "evolution_record.json",
                    "state": "physical_transaction_finalized_pending_generation",
                },
            )
        if args.execute and args.agent_mode == "model":
            memory_rollout = append_rollout_evidence(
                args.raw_attempt_memory,
                "",
                dry_run=False,
                policy=policy,
                report=report,
                trace=trace,
                checkpoint_before=checkpoint_before,
                checkpoint_after=effective_checkpoint_after,
                strategy_store=strategy_store,
                strategy_phase=_strategy_phase(
                    task,
                    evolution_runtime_state,
                    accepted_steps=episode_manifest.data.get("steps", ()),
                ),
                role_pipeline=role_pipeline,
                physically_dispatched=physically_dispatched,
                physical_transaction_completed=physical_transaction_completed,
            )
            _advance_physical_transaction_commit(
                transaction_commit_path,
                "memory_update",
                details={
                    "path": str(args.raw_attempt_memory),
                    "rollout_id": str(memory_rollout.get("rollout_id", "")),
                    "status": str(
                        memory_rollout.get("memory_update", {}).get("status", "")
                        if isinstance(memory_rollout.get("memory_update"), dict)
                        else ""
                    ),
                },
            )
            observation = replace(
                observation,
                retrieved_specs=tuple(
                    strategy_store.retrieve(
                        StrategyContext(
                            phase=_strategy_phase(
                                task,
                                evolution_runtime_state,
                                accepted_steps=episode_manifest.data.get("steps", ()),
                            ),
                            last_failure_type=(
                                "" if report.success else report.failure_type
                            ),
                            last_failure_level=(
                                "" if report.success else classify_report(report).level
                            ),
                        ),
                        limit=3,
                    )
                ),
            )
        try:
            if args.agent_mode == "model":
                generation_cycle = 0
                infeasible_scope_events: list[dict[str, object]] = []
                while True:
                    generation_cycle += 1
                    try:
                        controller_decision = evolver.decide_next_step(
                            policy,
                            report,
                            observation,
                            trace,
                            checkpoint_id=checkpoint_before,
                            physically_dispatched=physically_dispatched,
                            hard_iteration_limit_reached=(
                                (
                                    args.max_attempts > 0
                                    and attempt >= args.max_attempts
                                )
                            ),
                        )
                        write_json(
                            attempt_dir / "evolution_controller_decision.json",
                            controller_decision.to_dict(),
                        )
                        print(
                            f"[cap-box-carry] controller={controller_decision.action} "
                            f"physical_iterations={controller_decision.physical_iterations}/"
                            f"{controller_decision.max_physical_iterations or 'unlimited'} "
                            f"same_objective_failure_diagnostics="
                            f"{controller_decision.strategy_failure_count} "
                            "count_blocking=disabled",
                            flush=True,
                        )
                        if not controller_decision.should_evolve:
                            next_policy, spec = policy, ""
                        else:
                            next_policy, spec = evolver.evolve(
                                policy,
                                report,
                                observation,
                                trace,
                                controller_decision=controller_decision,
                            )
                        break
                    except Exception as generation_exc:
                        role_stall = _role_stall_from_exception(generation_exc)
                        if role_stall is None:
                            raise
                        stall_event = generation_stalls.record(
                            checkpoint_id=checkpoint_after or checkpoint_id,
                            generation_cycle=generation_cycle,
                            role=role_stall.role,
                            artifact_id=role_stall.artifact_id,
                            error=str(role_stall),
                            response_count=role_stall.response_count,
                            **_role_stall_record_fields(role_stall),
                        )
                        write_json(
                            attempt_dir
                            / f"role_generation_stall_cycle_{generation_cycle:02d}.json",
                            stall_event,
                        )
                        _publish_experiment_status(
                            role_pipeline,
                            "role_generation_stall",
                            attempt=attempt,
                            checkpoint_id=checkpoint_after or checkpoint_id,
                            policy=policy,
                            report=report,
                            trace=trace,
                            checkpoint_before=checkpoint_before,
                            checkpoint_after=checkpoint_after or checkpoint_id,
                            rollback_performed=rollback_performed,
                            next_system_operation=(
                                "re-observe the accepted checkpoint and restart "
                                "Orchestrator/Mutation/Coder without physical dispatch"
                            ),
                            artifact_dir=attempt_dir,
                            details=stall_event,
                        )
                        if stall_event.get("cycle_outcome") == MUTATION_SCOPE_INFEASIBLE:
                            infeasible_scope_events.append(stall_event)
                            if _all_recovery_levels_deterministically_infeasible(
                                infeasible_scope_events
                            ):
                                raise HostContractError(
                                    "all available recovery mutation levels were "
                                    "deterministically infeasible at checkpoint "
                                    f"{checkpoint_after or checkpoint_id}"
                                ) from generation_exc
                        if stall_event["system_stall"]:
                            raise SystemStallError(
                                "three complete generation cycles produced no "
                                "executable child at checkpoint "
                                f"{checkpoint_after or checkpoint_id}"
                            ) from generation_exc
                        evolver.reset_generation_cycle(
                            policy.attempt,
                            stall_event=stall_event,
                        )
                        refreshed_state = _checkpoint_authoritative_planning_state(
                            live_runtime_state=read_json(args.runtime_state_file),
                            checkpoint_reference_state=checkpoint_reference_state,
                            turn_episode=turn_episode,
                        )
                        refreshed_visual = camera.observe(
                            label=(
                                f"attempt_{attempt:02d}_generation_cycle_"
                                f"{generation_cycle + 1:02d}"
                            ),
                            runtime_state=refreshed_state,
                        )
                        if args.visual_evaluator == "model":
                            _require_visual_frame(
                                refreshed_visual,
                                label="generation recovery observation",
                            )
                        observation = replace(
                            observation,
                            runtime_state=refreshed_state,
                            visual_observations=(refreshed_visual,),
                        )
                        print(
                            "[cap-box-carry] "
                            + (
                                "MUTATION_SCOPE_INFEASIBLE; not model blame, but "
                                "this complete generation cycle is consumed; "
                                "re-invoking Evolution Manager"
                                if stall_event.get("cycle_outcome")
                                == MUTATION_SCOPE_INFEASIBLE
                                else (
                                    "ROLE_GENERATION_STALL; restarting the "
                                    "complete generation cycle "
                                    f"({stall_event['consecutive_stalls_at_checkpoint']}/"
                                    f"{generation_stalls.maximum_consecutive_stalls})"
                                )
                            )
                            + " without physical dispatch",
                            flush=True,
                        )
            else:
                controller_decision = evolution_controller.record_and_decide(
                    policy=policy,
                    report=report,
                    trace=trace,
                    checkpoint_id=checkpoint_before,
                    physically_dispatched=physically_dispatched,
                    hard_iteration_limit_reached=(
                        args.max_attempts > 0 and attempt >= args.max_attempts
                    ),
                )
                write_json(
                    attempt_dir / "evolution_controller_decision.json",
                    controller_decision.to_dict(),
                )
                if not controller_decision.should_evolve:
                    next_policy, spec = policy, ""
                else:
                    next_policy, spec = evolver.evolve(policy, report, observation)
        except Exception as exc:
            recovery_error: Exception | None = exc
            rewind_chain: list[dict[str, object]] = []
            while (
                args.execute
                and args.agent_mode == "model"
                and checkpoint_client is not None
                and controller_decision is not None
                and formal_allows_ancestor_rewind(host_contract_protocol)
                and _is_checkpoint_local_action_space_exhaustion(recovery_error)
            ):
                lineage = _checkpoint_parent_from_history(
                    checkpoint_id,
                    history_dirs=checkpoint_lineage_dirs,
                    accepted_manifest=episode_manifest.data,
                )
                parent_checkpoint_id = str(
                    lineage.get("parent_checkpoint_id", "")
                ).strip()
                if not parent_checkpoint_id:
                    break
                if not _checkpoint_parent_is_within_resume_root(
                    parent_checkpoint_id=parent_checkpoint_id,
                    resume_checkpoint_id=args.resume_checkpoint_id,
                ):
                    recovery_error = ModelAgentError(
                        "checkpoint-local action space exhausted at the explicit resume "
                        f"root {args.resume_checkpoint_id}; automatic rewind to ancestor "
                        f"{parent_checkpoint_id} is forbidden"
                    )
                    break
                retired_checkpoint_id = checkpoint_id
                blocked_objective = str(
                    lineage.get("creating_motion_objective", "")
                ).strip()
                lesson = (
                    f"Checkpoint {retired_checkpoint_id} is a verifier-confirmed local "
                    "dead end: every remaining host-valid coordination objective was "
                    "excluded by route evidence or bounded code validation. Return to "
                    f"its real parent {parent_checkpoint_id}. Do not recreate the retired "
                    "child transition; choose a distinct bounded geometry-changing "
                    "coordination primitive from current camera and physics evidence."
                )
                rewind_dir = (
                    attempt_dir
                    / "automatic_branch_rewind"
                    / f"{retired_checkpoint_id}_to_{parent_checkpoint_id}"
                )
                parent_heading_reference_rad = (
                    _straight_heading_reference_for_checkpoint(
                        parent_checkpoint_id,
                        references=straight_heading_references,
                        fallback_state=read_json(args.runtime_state_file),
                        runtime_state_file=args.runtime_state_file,
                    )
                )
                try:
                    retired_child_policy = _accepted_checkpoint_policy_from_manifest(
                        episode_manifest.data,
                        checkpoint_id=retired_checkpoint_id,
                    )
                    if retired_child_policy is None:
                        raise RuntimeError(
                            "accepted episode manifest does not contain the exact "
                            f"transaction that created {retired_checkpoint_id}"
                        )
                    retired_child_transaction_signature = (
                        _policy_transaction_signature(retired_child_policy)
                    )
                    retired_child_transaction = _policy_transaction_for_prompt(
                        retired_child_policy
                    )
                    _publish_experiment_status(
                        role_pipeline,
                        "rolling_back",
                        attempt=attempt,
                        checkpoint_id=retired_checkpoint_id,
                        policy=policy,
                        report=report,
                        trace=trace,
                        checkpoint_before=retired_checkpoint_id,
                        rollback_performed=True,
                        next_system_operation=(
                            f"restore trusted parent checkpoint {parent_checkpoint_id}"
                        ),
                        artifact_dir=rewind_dir,
                        details={"reason": str(recovery_error)},
                    )
                    rewind_verification = _restore_and_resynchronize(
                        checkpoint_client=checkpoint_client,
                        backend=backend,
                        checkpoint_id=parent_checkpoint_id,
                        checkpoint_reference_state={},
                        attempt_dir=rewind_dir,
                        heartbeat_file=args.heartbeat_file,
                        runtime_state_file=args.runtime_state_file,
                        idle_s=args.checkpoint_resync_idle_s,
                        idle_keepalive=idle_keepalive,
                        **ROLLBACK_POSE_LIMITS,
                        safety_profile=args.physical_safety_profile,
                        straight_heading_reference_rad=(
                            parent_heading_reference_rad
                        ),
                        video_recorder=action_video_recorder,
                        video_attempt=attempt,
                        video_kind="rewind",
                        video_label=f"branch_to_{parent_checkpoint_id}",
                        hold_planning=True,
                    )
                    rewound_state = read_json(args.runtime_state_file)
                    retired_source_artifacts = (
                        _manifest_source_artifacts_after_checkpoint(
                            episode_manifest.data,
                            parent_checkpoint_id,
                        )
                    )
                    retired_episode_steps = episode_manifest.rewind_to_checkpoint(
                        parent_checkpoint_id
                    )
                    retired_action_exemplars = action_memory.revoke_source_artifacts(
                        retired_source_artifacts
                    )
                    turn_episode = _reconstruct_turn_episode_from_manifest(
                        episode_manifest.data,
                        checkpoint_id=parent_checkpoint_id,
                    )
                    turn_episode = _reopen_turn_episode_after_branch_rewind(
                        turn_episode or empty_turn_episode(),
                        checkpoint_id=parent_checkpoint_id,
                        reason=lesson,
                        state_source=(
                            "accepted_episode_manifest_replay"
                            if turn_episode
                            else "empty_after_automatic_branch_rewind"
                        ),
                    )
                    write_json(turn_episode_path, turn_episode)
                    rewound_state = with_turn_episode(
                        rewound_state,
                        turn_episode,
                    )
                    branch_rewind_context = _build_branch_rewind_context(
                        task=task,
                        checkpoint_id=parent_checkpoint_id,
                        lesson=lesson,
                        runtime_state=rewound_state,
                        blocked_motion_objective=blocked_objective,
                        blocked_primitive_families=(policy_action,),
                        retired_child_transaction_signature=(
                            retired_child_transaction_signature
                        ),
                        retired_child_transaction=retired_child_transaction,
                    )
                    retired_failure_count = evolver.rewind_to_checkpoint(
                        parent_checkpoint_id,
                        branch_rewind_context=branch_rewind_context,
                    )
                except Exception as rewind_exc:
                    recovery_error = rewind_exc
                    break

                checkpoint_id = parent_checkpoint_id
                # The restored runtime state is the authoritative checkpoint
                # snapshot.  `checkpoint.reference_pose` is intentionally compact
                # pose evidence and omits lifecycle, route, contact, and formation
                # state; substituting it here corrupts the next planning context.
                checkpoint_reference_state = _checkpoint_reference_after_ancestor_rewind(
                    rewound_state
                )
                straight_heading_reference_rad = parent_heading_reference_rad
                _write_straight_heading_references(
                    straight_heading_ledger_path,
                    active_checkpoint_id=checkpoint_id,
                    references=straight_heading_references,
                )
                checkpoint_after = parent_checkpoint_id
                rollback_performed = True
                evolution_runtime_state = rewound_state
                _publish_experiment_status(
                    role_pipeline,
                    "rollback_complete",
                    attempt=attempt,
                    checkpoint_id=parent_checkpoint_id,
                    policy=policy,
                    report=report,
                    trace=trace,
                    checkpoint_before=retired_checkpoint_id,
                    checkpoint_after=parent_checkpoint_id,
                    rollback_performed=True,
                    next_system_operation=(
                        "re-plan the current subtask from the restored trusted parent"
                    ),
                    artifact_dir=rewind_dir,
                    details={"restore_verification": rewind_verification},
                )
                rewind_event = {
                    "attempt": attempt,
                    "retired_checkpoint_id": retired_checkpoint_id,
                    "restored_parent_checkpoint_id": parent_checkpoint_id,
                    "creating_motion_objective": blocked_objective,
                    "source_record": str(lineage.get("source_record", "")),
                    "trigger_error": str(recovery_error),
                    "retired_child_branch_failures": retired_failure_count,
                    "retired_episode_steps": retired_episode_steps,
                    "retired_action_exemplars": retired_action_exemplars,
                    "retired_source_artifacts": list(retired_source_artifacts),
                    "physical_iterations_preserved": (
                        evolution_controller.physical_iterations
                    ),
                    "restore_verification": rewind_verification,
                    "branch_rewind_context": branch_rewind_context,
                }
                rewind_chain.append(rewind_event)
                automatic_branch_rewinds.append(rewind_event)
                write_json(rewind_dir / "automatic_branch_rewind.json", rewind_event)
                write_json(
                    args.output_dir / "automatic_branch_rewinds.json",
                    automatic_branch_rewinds,
                )
                write_json(
                    args.output_dir / "branch_rewind_context.json",
                    branch_rewind_context,
                )
                print(
                    "[cap-box-carry] checkpoint-local action space exhausted; "
                    f"automatically rewound {retired_checkpoint_id} -> "
                    f"{parent_checkpoint_id} without another physical action",
                    flush=True,
                )

                rewind_visual = camera.observe(
                    label=(
                        f"attempt_{attempt:02d}_automatic_rewind_"
                        f"{parent_checkpoint_id}"
                    ),
                    runtime_state=rewound_state,
                )
                if args.visual_evaluator == "model":
                    _require_visual_frame(
                        rewind_visual,
                        label=f"automatic rewind to {parent_checkpoint_id}",
                    )
                failed_endpoint_visual, rewind_visual = (
                    _label_rollback_visual_timeline(post_visual, rewind_visual)
                )
                observation = replace(
                    observation,
                    runtime_state=rewound_state,
                    visual_observations=(failed_endpoint_visual, rewind_visual),
                    retrieved_specs=tuple(
                        strategy_store.retrieve(
                            StrategyContext(
                                phase=_strategy_phase(
                                    task,
                                    rewound_state,
                                    accepted_steps=episode_manifest.data.get("steps", ()),
                                ),
                                last_failure_type=report.failure_type,
                                last_failure_level=classify_report(report).level,
                            ),
                            limit=3,
                        )
                    ),
                )
                report = replace(
                    report,
                    evidence={
                        **report.evidence,
                        "automatic_branch_rewind": rewind_event,
                        "automatic_branch_rewind_chain": list(rewind_chain),
                    },
                )
                controller_decision = replace(
                    controller_decision,
                    action="switch_strategy",
                    should_evolve=True,
                    repair_level="strategy",
                    reason=(
                        "The current checkpoint exhausted its legal coordination action "
                        "space and was automatically rewound to a verified parent."
                    ),
                    required_change=(
                        "select a distinct geometry-changing coordination primitive at the "
                        "restored parent; do not recreate the retired child transition"
                    ),
                    strategy_signature="",
                    strategy_failure_count=0,
                )
                write_json(attempt_dir / "evaluation.json", report)
                write_json(
                    attempt_dir / "evolution_controller_decision.json",
                    controller_decision.to_dict(),
                )
                _persist_task_distribution(
                    args.output_dir,
                    policy,
                    attempt_dir=attempt_dir,
                    physical_dispatch_status="rejected_ancestor_checkpoint_restored",
                    robot_ports=robot_ports,
                    trace=trace,
                    report=report,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_after,
                    rollback_performed=True,
                )
                try:
                    next_policy, spec = evolver.evolve(
                        policy,
                        report,
                        observation,
                        trace,
                        controller_decision=controller_decision,
                    )
                except Exception as next_exc:
                    recovery_error = next_exc
                    continue
                recovery_error = None
                break

            if recovery_error is not None:
                interrupted_decision = _controller_decision_for_generation_error(
                    completed_decision=controller_decision,
                    policy=policy,
                    report=report,
                    trace=trace,
                    checkpoint_id=checkpoint_before,
                    physically_dispatched=physically_dispatched,
                    error=recovery_error,
                    evolution_controller=evolution_controller,
                )
                interrupted_record = EvolutionRecord(
                    task=task,
                    policy=policy,
                    observation=observation,
                    execution=trace,
                    evaluation=report,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=effective_checkpoint_after,
                    rollback_performed=rollback_performed,
                    controller_decision=interrupted_decision,
                )
                if transaction_record_index is None:
                    records.append(interrupted_record)
                else:
                    records[transaction_record_index] = interrupted_record
                _atomic_write_runner_json(
                    attempt_dir / "evolution_record.json", interrupted_record
                )
                write_json(
                    attempt_dir / "evolution_controller_decision.json",
                    interrupted_decision,
                )
                write_json(
                    attempt_dir / "model_evolution_error.json",
                    {
                        "error_type": type(recovery_error).__name__,
                        "error": str(recovery_error),
                        "resume_failed_attempt_dir": str(attempt_dir),
                        "automatic_branch_rewind_chain": rewind_chain,
                    },
                )
                _atomic_write_runner_jsonl(
                    args.output_dir / "evolution_records.jsonl", records
                )
                _advance_physical_transaction_commit(
                    transaction_commit_path,
                    "evolution_record",
                    details={
                        "path": "evolution_record.json",
                        "state": "generation_interrupted",
                        "error_type": type(recovery_error).__name__,
                    },
                )
                _publish_experiment_status(
                    role_pipeline,
                    "stopped_error",
                    attempt=attempt,
                    checkpoint_id=checkpoint_id,
                    policy=policy,
                    report=report,
                    trace=trace,
                    checkpoint_before=checkpoint_before,
                    checkpoint_after=checkpoint_after,
                    rollback_performed=rollback_performed,
                    next_system_operation="stop before another physical dispatch",
                    artifact_dir=attempt_dir,
                    details={
                        "error_type": type(recovery_error).__name__,
                        "error": str(recovery_error),
                    },
                )
                raise recovery_error
        record = EvolutionRecord(
            task=task,
            policy=policy,
            observation=observation,
            execution=trace,
            evaluation=report,
            evolved_spec=spec,
            next_script_id=(
                next_policy.script_id if controller_decision.should_evolve else ""
            ),
            checkpoint_before=checkpoint_before,
            checkpoint_after=effective_checkpoint_after,
            rollback_performed=rollback_performed,
            controller_decision=controller_decision.to_dict(),
        )
        if transaction_record_index is None:
            records.append(record)
        else:
            records[transaction_record_index] = record
        _atomic_write_runner_json(attempt_dir / "evolution_record.json", record)
        if args.write_html:
            write_evolution_html(args.output_dir / "evolution_report.html", records)
        _atomic_write_runner_jsonl(
            args.output_dir / "evolution_records.jsonl", records
        )
        _advance_physical_transaction_commit(
            transaction_commit_path,
            "evolution_record",
            details={
                "path": "evolution_record.json",
                "state": "complete",
                "next_script_id": record.next_script_id,
            },
        )
        if args.write_html:
            _refresh_human_review(args.output_dir)
        print(
            f"[cap-box-carry] attempt={attempt} script={policy.script_id} "
            f"step_accepted={report.success} task_complete={bool(report.metrics.get('task_complete', 0.0))} "
            f"failure={report.failure_type} level={classify_report(report).level} "
            f"output={attempt_dir}",
            flush=True,
        )
        if not controller_decision.should_evolve:
            print(
                f"[cap-box-carry] evolution stopped by host controller: "
                f"{controller_decision.action}; {controller_decision.reason}",
                flush=True,
            )
            _publish_experiment_status(
                role_pipeline,
                "task_complete" if task_complete else "stopped",
                attempt=attempt,
                checkpoint_id=checkpoint_after or checkpoint_id,
                policy=policy,
                report=report,
                trace=trace,
                checkpoint_before=checkpoint_before,
                checkpoint_after=checkpoint_after or checkpoint_id,
                rollback_performed=rollback_performed,
                next_system_operation=(
                    "none; task completion is verifier-confirmed"
                    if task_complete
                    else "operator review is required before another physical dispatch"
                ),
                artifact_dir=attempt_dir,
                details={"controller_decision": controller_decision.to_dict()},
            )
            break
        if report.failure_type == "runtime_state_missing":
            print(
                "[cap-box-carry] runtime_state missing; stop here because this needs live Term1 state, not script evolution.",
                flush=True,
            )
            _publish_experiment_status(
                role_pipeline,
                "stopped_error",
                attempt=attempt,
                checkpoint_id=checkpoint_id,
                policy=policy,
                report=report,
                trace=trace,
                checkpoint_before=checkpoint_before,
                checkpoint_after=checkpoint_after,
                rollback_performed=rollback_performed,
                next_system_operation="restore the live Term1 runtime before resuming",
                artifact_dir=attempt_dir,
            )
            break
        policy = next_policy
        _print_team_assignment(policy)
        _persist_task_distribution(
            args.output_dir,
            policy,
            robot_ports={"front": backend.front_port, "back": backend.back_port},
        )

    finalize_run()
    _ACTIVE_RUN_FINALIZER = None
    print(f"[cap-box-carry] wrote {args.output_dir / 'final_summary.json'}", flush=True)


def main() -> None:
    """Run one experiment and seal any started formal run on terminal errors."""

    global _ACTIVE_RUN_FINALIZER
    try:
        _run_main()
    except BaseException as exc:
        context = _ACTIVE_RUN_FINALIZER or {}
        finalize = context.get("finalize")
        if callable(finalize):
            try:
                finalize(
                    status=_terminal_reason_for_exception(exc),
                    terminal_error=exc,
                )
            except Exception as finalization_exc:
                print(
                    "[cap-box-carry] failed to seal terminal run artifacts: "
                    f"{type(finalization_exc).__name__}: {finalization_exc}",
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        _ACTIVE_RUN_FINALIZER = None


def _terminal_reason_for_exception(exc: BaseException) -> str:
    """Map every terminal exception to the frozen v3 reason vocabulary."""

    if isinstance(exc, KeyboardInterrupt):
        # An operator stop says nothing about either model competence or backend
        # health.  Keep it outside the formal result set without manufacturing a
        # simulator failure.
        return "OPERATOR_CANCELLED"
    if isinstance(exc, ProviderFatalError):
        return "PROVIDER_FATAL"
    if isinstance(exc, HostContractError):
        return "HOST_CONTRACT_ERROR"
    if isinstance(exc, CandidateCheckpointRejectedError):
        # Candidate drift is a local reject; escaping to the finalizer means the
        # Host failed to restore the trusted parent after discarding it.
        return "BACKEND_FATAL"
    if isinstance(exc, SystemStallError) or _role_stall_from_exception(exc):
        return "SYSTEM_STALL"
    if isinstance(
        exc,
        (CheckpointBackendFatalError, TimeoutError, ConnectionError, OSError),
    ):
        return "BACKEND_FATAL"
    # Unexpected model/contract failures are system failures, never task failures.
    if isinstance(exc, (ModelAgentError, ModelRoleError)):
        return "SYSTEM_STALL"
    return "BACKEND_FATAL"


def _controller_decision_for_generation_error(
    *,
    completed_decision: EvolutionDecision | None,
    policy: PolicyScript,
    report: EvaluationReport,
    trace: ExecutionTrace,
    checkpoint_id: str,
    physically_dispatched: bool,
    error: BaseException,
    evolution_controller: EvolutionController,
) -> dict[str, object]:
    """Seal the current physical result without borrowing an older decision."""

    decision = completed_decision or evolution_controller.record_and_decide(
        policy=policy,
        report=report,
        trace=trace,
        checkpoint_id=checkpoint_id,
        physically_dispatched=physically_dispatched,
    )
    payload = decision.to_dict()
    payload.update(
        {
            "generation_interrupted": True,
            "next_policy_generated": False,
            "generation_error_type": type(error).__name__,
            "generation_error": str(error),
        }
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-protocol",
        choices=(FORMAL_EVAL_PROTOCOL_ID,),
        default="",
        help=(
            "Enable the immutable formal benchmark protocol. Omit for engineering "
            "and qualitative development runs."
        ),
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help=f"Zero-based formal repetition index; {FORMAL_EVAL_PROTOCOL_ID} uses 0..4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Formal run seed. The protocol binds it to --run-index.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=("model", "template"),
        default="model",
        help="model is the formal self-evolution path; template is smoke-test/ablation only.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--planner-backend",
        choices=("local_qwen_vl", "openai_compatible"),
        default="local_qwen_vl",
        help=(
            "Shared multimodal role backend. local_qwen_vl is the engineering/offline "
            f"path; {FORMAL_EVAL_PROTOCOL_ID} requires openai_compatible with one frozen exact model ID."
        ),
    )
    parser.add_argument("--planner-model", default="gpt-5.4-mini")
    parser.add_argument("--planner-base-url", default=os.environ.get("COEVOLVE_BASE_URL", ""))
    parser.add_argument("--planner-api-key-env", default="COEVOLVE_API_KEY")
    parser.add_argument("--planner-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--planner-network-retries",
        type=int,
        default=2,
        help=(
            "Number of retries for transient provider transport failures. Formal "
            "evaluation overrides this with its frozen infrastructure policy."
        ),
    )
    parser.add_argument(
        "--keep-term4-on-transient-provider-error",
        action="store_true",
        help=(
            "Keep the active model call alive after a complete retry round of "
            "HTTP 408/409/429/5xx or transport failures. Non-retryable provider, "
            "identity, contract, and model-output errors still terminate normally."
        ),
    )
    parser.add_argument(
        "--provider-recovery-backoff-s",
        type=float,
        default=60.0,
        help="Delay between complete transient-provider retry rounds.",
    )
    parser.add_argument(
        "--provider-recovery-wall-time-cap-s",
        type=float,
        default=0.0,
        help=(
            "Maximum wall time for one unchanged logical call across transient "
            "provider recovery rounds; zero disables the cap outside formal runs."
        ),
    )
    parser.add_argument(
        "--allow-dirty-codebase",
        action="store_true",
        help=(
            "Engineering-only override for a model+execute run. The exact dirty "
            "source snapshot is still recorded and must remain unchanged."
        ),
    )
    parser.add_argument(
        "--diagnostic-formal-host-contracts",
        action="store_true",
        help=(
            "Engineering-only continuation switch: use the current formal Host "
            "contracts and exact resolved-model check without claiming a formal run. "
            "Requires a resume source and an output directory under non_paper."
        ),
    )
    parser.add_argument(
        "--visual-evaluator",
        choices=("model", "none"),
        default="model",
        help="model uses Qwen-VL for image-based evaluator evidence; none is for ablation/debug only.",
    )
    parser.add_argument("--vl-model-name", default=DEFAULT_MULTIMODAL_MODEL)
    parser.add_argument(
        "--preload-vl-model",
        action="store_true",
        help="Preload the single shared Qwen3-VL model before the experiment.",
    )
    parser.add_argument("--vl-device-map", default="auto")
    parser.add_argument("--vl-max-gpu-memory", default=None)
    parser.add_argument("--vl-max-cpu-memory", default=None)
    parser.add_argument(
        "--vl-quantization",
        choices=("none", "8bit", "4bit"),
        default="none",
        help=(
            "Local Qwen3-VL weight mode. The engineering default uses normal "
            "BF16/auto-dtype; 8bit and 4bit are low-memory options. This setting is "
            "unused by the remote formal backbone benchmark."
        ),
    )
    parser.add_argument("--vl-temperature", type=float, default=0.0)
    parser.add_argument("--vl-max-new-tokens", type=int, default=1536)
    parser.add_argument("--task-id", default="two_g1_long_box_carry_manual_layout")
    parser.add_argument(
        "--route-spec",
        type=Path,
        default=REPO_ROOT / "configs" / "routes" / "two_g1_box_push_layout.json",
        help=(
            "Scene-authored connected-route geometry. It cannot prescribe skills, "
            "roles, synchronization modes, turn signs, or action magnitudes."
        ),
    )
    parser.add_argument(
        "--target",
        default=(
            "Two G1 robots carry the long_box through the authored scene and "
            "stably place it on the red target cylinder."
        ),
    )
    current_run_dir = REPO_ROOT / "results" / "deployment" / "cap_box_carry_live"
    parser.add_argument("--output-dir", type=Path, default=current_run_dir)
    parser.add_argument(
        "--runtime-state-file",
        type=Path,
        default=current_run_dir / "runtime_state.json",
    )
    parser.add_argument(
        "--heartbeat-file",
        type=Path,
        default=current_run_dir / "planner_heartbeat.json",
    )
    parser.add_argument(
        "--raw-attempt-memory",
        type=Path,
        default=None,
        help=(
            "Append-only verifier/model rollout evidence. This file is never injected directly "
            "into the Agent prompt."
        ),
    )
    parser.add_argument(
        "--strategy-memory",
        type=Path,
        default=None,
        help="Compact verifier-grounded deployment decision rules retrieved by the Agent.",
    )
    parser.add_argument(
        "--action-exemplar-memory",
        type=Path,
        default=None,
        help=(
            "Verifier-accepted executable examples retrieved only after method, semantic "
            "direction, and coordination mode have been selected."
        ),
    )
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        default=None,
        help=(
            "Contiguous accepted-checkpoint script manifest. Default: "
            "<output-dir>/accepted_episode_manifest.json."
        ),
    )
    parser.add_argument(
        "--spec-memory",
        type=Path,
        default=None,
        help="Deprecated alias for --raw-attempt-memory.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help=(
            "Largest attempt number to execute; a resumed attempt continues through this "
            "number. Use 0 for no attempt limit."
        ),
    )
    parser.add_argument(
        "--iteration-budget",
        type=int,
        default=20,
        help=(
            "Hard maximum number of verifier-evaluated physical rollouts in one evolution run. "
            "Model-only JSON corrections do not consume this budget. Use 0 for no physical "
            "iteration limit."
        ),
    )
    parser.add_argument(
        "--strategy-failure-budget",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option. Physical failure counts are diagnostic only "
            "and never block a strategy or semantic objective."
        ),
    )
    parser.add_argument(
        "--resume-policy",
        type=Path,
        default=None,
        help="Resume an already generated team_plan.json/policy.json without regenerating attempt 1.",
    )
    parser.add_argument(
        "--resume-checkpoint-id",
        default=None,
        help=(
            "Restore an accepted_step_N checkpoint already held by the same live Term1 "
            "runtime and continue numbering from N. Use with --execute; checkpoints do "
            "not survive a Term1 restart."
        ),
    )
    parser.add_argument(
        "--next-accepted-step-index",
        type=int,
        default=None,
        help=(
            "Explicit next checkpoint number after a resumed state. Use 39 when "
            "accepted_step_38 is revoked; directory contents never choose this number."
        ),
    )
    parser.add_argument(
        "--next-policy-attempt-index",
        type=int,
        default=None,
        help=(
            "Explicit next policy attempt number when resuming an older artifact into a "
            "new result directory. This prevents script/evidence identity collisions."
        ),
    )
    parser.add_argument(
        "--resume-failed-attempt-dir",
        type=Path,
        default=None,
        help=(
            "Evolve a persisted rejected attempt into the next policy without physically "
            "replaying the failed action."
        ),
    )
    parser.add_argument(
        "--resume-attempt-dir",
        type=Path,
        default=None,
        help=(
            "Evolve any persisted attempt into its next policy without replaying it. "
            "Use this after a model/network interruption that occurred after an accepted step."
        ),
    )
    parser.add_argument(
        "--branch-rewind-note",
        default="",
        help=(
            "Verifier-grounded lesson from an explicitly abandoned descendant branch. "
            "It temporarily opens the strategy family at --resume-checkpoint-id and "
            "expires after the next accepted checkpoint."
        ),
    )
    parser.add_argument(
        "--branch-rewind-route-exclusion",
        default="",
        help=(
            "Engineering-only exact checkpoint-local route exclusion in "
            "primitive:semantic_objective form. Use only with --branch-rewind-note "
            "when measured descendant evidence proves route regression."
        ),
    )
    parser.add_argument(
        "--checkpoint-timeout-s",
        type=float,
        default=10.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-resync-settle-s",
        type=float,
        default=4.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-resync-idle-s",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--initial-support-timeout-s",
        type=float,
        default=120.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--initial-support-stable-s",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--task-stage",
        choices=("stage1_straight", "stage2_corridor", "stage3_right", "full_scene"),
        default="full_scene",
    )
    parser.add_argument(
        "--route-discovery",
        dest="route_discovery",
        action="store_true",
        help=(
            "Let the model discover intermediate route geometry from camera and rollout "
            "evidence; no host waypoint coordinates or waypoint-distance reward are exposed."
        ),
    )
    parser.add_argument(
        "--host-waypoint-guidance",
        dest="route_discovery",
        action="store_false",
        help="Legacy ablation that exposes stage waypoint vectors to the model.",
    )
    parser.set_defaults(route_discovery=True)
    parser.add_argument("--execute", action="store_true", help="Actually publish static skill commands. Default only writes scripts/commands.")
    parser.add_argument(
        "--stop-before-physical-dispatch",
        action="store_true",
        help=(
            "With --execute, restore live state and validate the generated policy, "
            "then stop immediately before publishing Term4 commands."
        ),
    )
    parser.add_argument("--write-html", action="store_true", help="Also write the optional offline HTML report.")
    parser.add_argument("--start-pulse-seconds", type=float, default=0.0)
    parser.add_argument("--stand-seconds", type=float, default=0.5)
    parser.add_argument(
        "--bootstrap-start-pulse-seconds",
        type=float,
        default=4.0,
        help="Initial dual-planner START/IDLE window used before the Agent places the deferred payload.",
    )
    parser.add_argument(
        "--bootstrap-idle-seconds",
        type=float,
        default=6.0,
        help="Additional synchronized IDLE hold used to finish drop, balance, and arm-lock ramp.",
    )
    parser.add_argument(
        "--transport-skill",
        choices=("carry_lateral_left", "carry_lateral_right", "carry_forward", "carry_backward"),
        default="carry_forward",
        help=(
            "Nominal first-call seed exposed to the model. It is advisory; route and formation "
            "must be inferred from observations and verifier evidence."
        ),
    )
    parser.add_argument("--transport-distance-m", type=float, default=1.0)
    parser.add_argument("--transport-speed-mps", type=float, default=0.50)
    parser.add_argument(
        "--minimum-transport-window-s",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--minimum-lateral-transport-window-s",
        type=float,
        default=0.4,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--transport-duration-gain",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--transport-duration-baseline-m",
        type=float,
        default=2.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--minimum-transport-ramp-s",
        type=float,
        default=1.5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--post-transport-idle-s",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--transport-ramp-s",
        type=float,
        default=1.5,
        help="Requested IDLE settle before and after movement; executor enforces its safety floor.",
    )
    parser.add_argument("--forward-distance-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--forward-speed-mps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--physical-safety-profile",
        "--payload-safety-profile",
        dest="physical_safety_profile",
        choices=tuple(CARRY_SAFETY_PROFILES),
        default="task",
        help=(
            "Single physical verifier envelope. task only rolls back true loss of "
            "the payload/support/upright state; strict is a legacy debug profile."
        ),
    )
    parser.add_argument(
        "--post-action-stable-s",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--post-action-stability-timeout-s",
        type=float,
        default=5.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--precommit-stable-s",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--precommit-stability-timeout-s",
        type=float,
        default=5.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-team-sync", action="store_true", help="Disable live dual-root feedback for ablation only.")
    parser.add_argument(
        "--team-sync-active-catchup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use low-rate root-pose feedback to apply smooth bounded leader/lagger "
            "speed corrections during synchronized loaded translation. Enabled for "
            "formal evolution runs; use --no-team-sync-active-catchup for ablation."
        ),
    )
    parser.add_argument("--team-sync-longitudinal-kp", type=float, default=2.0, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-cross-track-kp", type=float, default=0.6, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-payload-kp", type=float, default=1.2, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-error-filter-alpha", type=float, default=0.35, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-pause-leader-error-m", type=float, default=0.008, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-resume-error-m", type=float, default=0.003, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-lagging-boost-mps", type=float, default=0.06, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-max-correction-mps", type=float, default=0.06, help=argparse.SUPPRESS)
    parser.add_argument("--team-sync-min-active-speed-ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--no-require-payload-on-target", action="store_true", help="Disable final target placement success criterion for stage-only ablations.")
    parser.add_argument(
        "--require-robots-exit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require the ordered post-release lifecycle: synchronized crouched "
            "exit_backward, recover_stand, then upright robot clearance from the box."
        ),
    )
    parser.add_argument("--camera-device", type=int, default=None)
    parser.add_argument("--visual-frame", type=Path, default=None)
    parser.add_argument(
        "--record-action-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Record fixed-global MP4 clips only during physical actions and "
            "checkpoint restores, then concatenate the physical timeline."
        ),
    )
    parser.add_argument(
        "--action-video-fps",
        type=float,
        default=10.0,
        help="Frame rate for action/rollback clips from the fixed global camera.",
    )
    parser.add_argument(
        "--visual-window-name",
        default="MuJoCo : praxis_manual_layout_two_g1_43dof_scene",
        help="Capture this X11 window each attempt and pass the screenshot to Qwen-VL.",
    )
    args = parser.parse_args()
    if args.raw_attempt_memory is not None and args.spec_memory is not None:
        parser.error("use --raw-attempt-memory; do not also pass deprecated --spec-memory")
    if args.max_attempts < 0:
        parser.error("--max-attempts must be non-negative")
    if args.iteration_budget < 0:
        parser.error("--iteration-budget must be non-negative")
    if args.strategy_failure_budget < 0:
        parser.error("--strategy-failure-budget must be non-negative")
    args.raw_attempt_memory = args.raw_attempt_memory or args.spec_memory
    if args.forward_distance_m is not None:
        args.transport_distance_m = args.forward_distance_m
    if args.forward_speed_mps is not None:
        args.transport_speed_mps = args.forward_speed_mps
    if args.team_sync_pause_leader_error_m <= args.team_sync_resume_error_m:
        parser.error("team sync pause threshold must exceed its resume threshold")
    if args.stop_before_physical_dispatch and not args.execute:
        parser.error("--stop-before-physical-dispatch requires --execute")
    if args.resume_checkpoint_id is not None:
        if not args.execute:
            parser.error("--resume-checkpoint-id requires --execute")
        try:
            _accepted_checkpoint_index(args.resume_checkpoint_id)
        except ValueError as exc:
            parser.error(str(exc))
        if (
            args.next_accepted_step_index is not None
            and args.next_accepted_step_index
            <= _accepted_checkpoint_index(args.resume_checkpoint_id)
        ):
            parser.error(
                "--next-accepted-step-index must be greater than the resumed checkpoint index"
            )
    elif args.next_accepted_step_index is not None:
        parser.error("--next-accepted-step-index requires --resume-checkpoint-id")
    if args.resume_attempt_dir is not None and args.resume_failed_attempt_dir is not None:
        parser.error("use only one of --resume-attempt-dir and --resume-failed-attempt-dir")
    if args.next_policy_attempt_index is not None:
        if args.next_policy_attempt_index < 1:
            parser.error("--next-policy-attempt-index must be positive")
        if (
            args.resume_attempt_dir is None
            and args.resume_failed_attempt_dir is None
            and args.resume_checkpoint_id is None
        ):
            parser.error(
                "--next-policy-attempt-index requires --resume-checkpoint-id, "
                "--resume-attempt-dir, or --resume-failed-attempt-dir"
            )
    if args.branch_rewind_note and args.resume_checkpoint_id is None:
        parser.error("--branch-rewind-note requires --resume-checkpoint-id")
    if args.branch_rewind_route_exclusion and not args.branch_rewind_note:
        parser.error(
            "--branch-rewind-route-exclusion requires --branch-rewind-note"
        )
    if args.branch_rewind_route_exclusion:
        method, separator, objective = str(
            args.branch_rewind_route_exclusion
        ).strip().partition(":")
        if (
            not separator
            or method not in COORDINATION_PRIMITIVE_METHODS
            or not objective
        ):
            parser.error(
                "--branch-rewind-route-exclusion must be an exact "
                "coordination primitive:semantic_objective"
            )
    if args.diagnostic_formal_host_contracts:
        if args.eval_protocol:
            parser.error(
                "--diagnostic-formal-host-contracts cannot be combined with "
                "--eval-protocol"
            )
        if (
            args.resume_checkpoint_id is None
            or (
                args.resume_attempt_dir is None
                and args.resume_failed_attempt_dir is None
                and args.resume_policy is None
            )
        ):
            parser.error(
                "--diagnostic-formal-host-contracts requires a checkpoint and "
                "persisted resume source"
            )
        non_paper_root = (
            REPO_ROOT / "results" / "deployment" / "non_paper"
        ).resolve(strict=False)
        try:
            args.output_dir.resolve(strict=False).relative_to(non_paper_root)
        except ValueError:
            parser.error(
                "--diagnostic-formal-host-contracts requires --output-dir under "
                f"{non_paper_root}"
            )
    if not 0.0 <= args.team_sync_min_active_speed_ratio <= 1.0:
        parser.error("--team-sync-min-active-speed-ratio must be within [0, 1]")
    if not 0.0 <= args.team_sync_error_filter_alpha <= 1.0:
        parser.error("--team-sync-error-filter-alpha must be within [0, 1]")
    if min(
        args.post_action_stable_s,
        args.post_action_stability_timeout_s,
        args.precommit_stable_s,
        args.precommit_stability_timeout_s,
    ) < 0.0:
        parser.error("post-action stability durations must be non-negative")
    try:
        configure_formal_args(
            args,
            repo_root=REPO_ROOT,
            default_output_dir=current_run_dir,
        )
    except FormalEvalConfigurationError as exc:
        parser.error(str(exc))
    return args


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resume_history_output_dirs(resume_attempt_dir: Path | None) -> tuple[Path, ...]:
    """Follow resume_source links so verifier failures survive process restarts."""

    if resume_attempt_dir is None:
        return ()
    current_attempt = Path(resume_attempt_dir).expanduser()
    if not current_attempt.is_absolute():
        current_attempt = REPO_ROOT / current_attempt
    output_dirs: list[Path] = []
    seen: set[Path] = set()
    while True:
        output_dir = current_attempt.resolve(strict=False).parent
        if output_dir in seen:
            break
        seen.add(output_dir)
        output_dirs.append(output_dir)
        source = read_json(output_dir / "resume_source.json")
        parent_attempt = str(source.get("resume_attempt_dir", "")).strip()
        if not parent_attempt:
            break
        current_attempt = Path(parent_attempt).expanduser()
        if not current_attempt.is_absolute():
            current_attempt = REPO_ROOT / current_attempt
    return tuple(reversed(output_dirs))


def _resume_model_evidence_history_dirs(
    resume_attempt_dir: Path | None,
    *,
    resume_history_dirs: tuple[Path, ...],
    checkpoint_id: str,
    accepted_manifest: dict[str, object] | None,
) -> tuple[Path, ...]:
    """Keep hard failures from the current physical checkpoint incarnation."""

    if resume_attempt_dir is None:
        return ()
    direct_output_dir = (
        Path(resume_attempt_dir).expanduser().resolve(strict=False).parent
    )
    history_dirs = tuple(
        Path(path).expanduser().resolve(strict=False) for path in resume_history_dirs
    )
    steps = (
        accepted_manifest.get("steps", ())
        if isinstance(accepted_manifest, dict)
        else ()
    )
    checkpoint_origin: Path | None = None
    if isinstance(steps, (list, tuple)):
        for step in reversed(steps):
            if (
                not isinstance(step, dict)
                or str(step.get("checkpoint_after", "")).strip() != checkpoint_id
            ):
                continue
            source_artifact_dir = str(step.get("source_artifact_dir", "")).strip()
            if source_artifact_dir:
                checkpoint_origin = (
                    Path(source_artifact_dir)
                    .expanduser()
                    .resolve(strict=False)
                    .parent
                )
            break
    if checkpoint_origin is not None and checkpoint_origin in history_dirs:
        origin_index = history_dirs.index(checkpoint_origin)
        return history_dirs[origin_index:]
    # Without a manifest anchor, avoid importing a physically different older
    # checkpoint that happens to reuse the same accepted_step_N identifier.
    return (direct_output_dir,)


def _resume_task_plan_from_attempt(
    resume_attempt_dir: Path | None,
) -> tuple[dict[str, object], Path | None]:
    """Recover the exact attempt plan, falling back through its resume lineage."""

    if resume_attempt_dir is None:
        return {}, None
    attempt_dir = Path(resume_attempt_dir).expanduser()
    if not attempt_dir.is_absolute():
        attempt_dir = REPO_ROOT / attempt_dir
    policy_payload = read_json(attempt_dir / "policy.json")
    if not policy_payload:
        policy_payload = read_json(attempt_dir / "team_plan.json")
    try:
        attempt = int(policy_payload.get("attempt"))
    except (TypeError, ValueError):
        try:
            attempt = int(attempt_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            return {}, None
    role_dir = attempt_dir.parent / "model_roles"
    candidates = (
        role_dir / f"task_orchestrator_plan_attempt_{attempt:02d}.json",
        role_dir / f"task_orchestrator_plan_attempt_{attempt}.json",
    )
    for path in candidates:
        payload = read_json(path)
        if isinstance(payload, dict) and payload:
            return payload, path

    evaluation = read_json(attempt_dir / "evaluation.json")
    if evaluation and not bool(evaluation.get("success", False)):
        resume_source = read_json(attempt_dir.parent / "resume_source.json")
        accepted_parent = str(
            resume_source.get("resume_attempt_dir", "")
        ).strip()
        if accepted_parent:
            parent_attempt_dir = Path(accepted_parent).expanduser()
            if not parent_attempt_dir.is_absolute():
                parent_attempt_dir = REPO_ROOT / parent_attempt_dir
            if parent_attempt_dir.resolve(strict=False) != attempt_dir.resolve(
                strict=False
            ):
                return _resume_task_plan_from_attempt(parent_attempt_dir)
    return {}, None


def _resume_rejected_primitive_family(
    resume_attempt_dir: Path | None,
) -> str:
    """Recover the verifier-rejected method family when resuming an old branch."""

    if resume_attempt_dir is None:
        return ""
    attempt_dir = Path(resume_attempt_dir)
    report = read_json(attempt_dir / "evaluation.json")
    if str(report.get("failure_type", "")) not in {
        "route_regression",
        "alignment_regression",
    }:
        return ""
    try:
        policy = _load_policy(attempt_dir / "policy.json")
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return ""
    method = _policy_action_method(policy)
    return method if method in COORDINATION_PRIMITIVE_METHODS else ""


def _checkpoint_parent_from_history(
    checkpoint_id: str,
    *,
    history_dirs: tuple[Path, ...],
    accepted_manifest: dict[str, object] | None = None,
) -> dict[str, str]:
    """Recover one checkpoint's real parent from accepted evolution records."""

    checkpoint_id = str(checkpoint_id).strip()
    if not checkpoint_id:
        return {}
    manifest_steps = (
        accepted_manifest.get("steps", ())
        if isinstance(accepted_manifest, dict)
        else ()
    )
    if isinstance(manifest_steps, (list, tuple)):
        for step in reversed(manifest_steps):
            if not isinstance(step, dict):
                continue
            checkpoint_after = str(step.get("checkpoint_after", "")).strip()
            checkpoint_before = str(step.get("checkpoint_before", "")).strip()
            if (
                checkpoint_after == checkpoint_id
                and checkpoint_before
                and checkpoint_before != checkpoint_after
            ):
                source_dir = str(step.get("source_artifact_dir", "")).strip()
                return {
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": checkpoint_before,
                    "creating_motion_objective": _manifest_motion_objective(step),
                    "source_record": (
                        str(Path(source_dir) / "evolution_record.json")
                        if source_dir
                        else "accepted_episode_manifest"
                    ),
                }
    for output_dir in reversed(tuple(dict.fromkeys(history_dirs))):
        output_dir = Path(output_dir)
        attempt_dirs = sorted(
            output_dir.glob("attempt_*"),
            key=_attempt_directory_index,
            reverse=True,
        )
        for attempt_dir in attempt_dirs:
            record_path = attempt_dir / "evolution_record.json"
            record = read_json(record_path)
            checkpoint_after = str(record.get("checkpoint_after", "")).strip()
            checkpoint_before = str(record.get("checkpoint_before", "")).strip()
            evaluation = record.get("evaluation", {})
            accepted = bool(
                evaluation.get("success", False)
                if isinstance(evaluation, dict)
                else False
            )
            if (
                accepted
                and checkpoint_after == checkpoint_id
                and checkpoint_before
                and checkpoint_before != checkpoint_after
            ):
                return {
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": checkpoint_before,
                    "creating_motion_objective": _record_motion_objective(record),
                    "source_record": str(record_path),
                }
    return {}


def _manifest_motion_objective(step: dict[str, object]) -> str:
    robot_calls = step.get("robot_calls", {})
    if not isinstance(robot_calls, dict):
        return ""
    for robot_id in ("front", "back"):
        call = robot_calls.get(robot_id, {})
        if not isinstance(call, dict):
            continue
        method = str(call.get("method", "")).strip()
        parameters = call.get("parameters", {})
        if method not in COORDINATION_PRIMITIVE_METHODS or not isinstance(
            parameters, dict
        ):
            continue
        semantic_parameter = {
            "coordinated_linear_transport": "direction",
            "coordinated_arc_transport": "turn_direction",
            "coordinated_pivot": "turn_direction",
            "coordinated_alignment": "axis",
        }.get(method, "")
        semantic_value = str(parameters.get(semantic_parameter, "")).strip()
        return f"{method}:{semantic_value}" if semantic_value else method
    return ""


def _attempt_directory_index(path: Path) -> int:
    try:
        return int(Path(path).name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _record_motion_objective(record: dict) -> str:
    policy = record.get("policy", {})
    if not isinstance(policy, dict):
        return ""
    assignments = policy.get("assignments", ())
    if not isinstance(assignments, (list, tuple)):
        return ""
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        objective = str(assignment.get("objective", "")).strip()
        method, separator, semantic_objective = objective.partition(":")
        if (
            separator
            and method in COORDINATION_PRIMITIVE_METHODS
            and semantic_objective
        ):
            return objective
    return ""


def _accepted_checkpoint_policy_from_manifest(
    manifest: dict,
    *,
    checkpoint_id: str,
) -> PolicyScript | None:
    """Rebuild the exact team transaction that created one accepted child."""

    checkpoint_id = str(checkpoint_id).strip()
    if not checkpoint_id or not isinstance(manifest, dict):
        return None
    for step in reversed(tuple(manifest.get("steps", ()))):
        if (
            not isinstance(step, dict)
            or str(step.get("checkpoint_after", "")).strip() != checkpoint_id
        ):
            continue
        scripts_payload = step.get("robot_scripts", {})
        if not isinstance(scripts_payload, dict):
            return None
        robot_scripts: list[RobotPolicyScript] = []
        for robot_id in ("front", "back"):
            item = scripts_payload.get(robot_id, {})
            if not isinstance(item, dict):
                return None
            code = str(item.get("code", "")).strip()
            if not code:
                return None
            robot_scripts.append(
                RobotPolicyScript(
                    robot_id=robot_id,
                    script_id=str(
                        item.get("script_id")
                        or f"{robot_id}_{step.get('script_id', checkpoint_id)}"
                    ),
                    code=code,
                    synchronization_group=str(
                        item.get("synchronization_group", "accepted_team")
                    ),
                    phase_id=str(item.get("phase_id", "")),
                    contact_intent=str(
                        item.get("contact_intent", "unspecified")
                    ),
                    coordination_mode=_persisted_coordination_mode(
                        item.get(
                            "coordination_mode",
                            step.get("coordination_mode", "sync"),
                        )
                    ),
                )
            )
        return PolicyScript(
            script_id=str(step.get("script_id") or checkpoint_id),
            attempt=max(1, int(step.get("attempt", 1))),
            code=robot_scripts[0].code,
            robot_scripts=tuple(robot_scripts),
            verification_contract=dict(step.get("verification_contract", {})),
        )
    return None


def _accepted_child_policy_after_checkpoint(
    manifest: dict,
    *,
    checkpoint_id: str,
) -> PolicyScript | None:
    """Rebuild the first accepted child retired by an explicit ancestor rewind."""

    checkpoint_id = str(checkpoint_id).strip()
    if not checkpoint_id or not isinstance(manifest, dict):
        return None
    for step in tuple(manifest.get("steps", ())):
        if (
            isinstance(step, dict)
            and str(step.get("checkpoint_before", "")).strip() == checkpoint_id
        ):
            child_checkpoint_id = str(step.get("checkpoint_after", "")).strip()
            return _accepted_checkpoint_policy_from_manifest(
                manifest,
                checkpoint_id=child_checkpoint_id,
            )
    return None


def _is_checkpoint_local_action_space_exhaustion(exc: Exception) -> bool:
    if not isinstance(exc, ModelAgentError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in CHECKPOINT_LOCAL_EXHAUSTION_MARKERS)


def _role_stall_from_exception(
    exc: BaseException,
) -> ModelRoleStallError | None:
    """Find a role-stall cause through ModelAgentError wrapper boundaries."""

    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ModelRoleStallError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _role_stall_record_fields(stall: ModelRoleStallError) -> dict[str, object]:
    """Project typed role evidence into the checkpoint-local stall audit."""

    return {
        "cycle_outcome": stall.cycle_outcome,
        "selected_mutation_level": stall.selected_mutation_level,
        "validation_errors": stall.validation_errors,
        "validation_error_kinds": stall.validation_error_kinds,
        "typed_failure": stall.typed_failure,
        "attempted_change": stall.attempted_change,
        "editable_fields": stall.editable_fields,
        "result": stall.result,
    }


def _record_formal_shadow_physical_result(
    output_dir: Path,
    *,
    attempt: int,
    report: EvaluationReport,
    trace: ExecutionTrace,
) -> None:
    """Join shadow Host advice with the actual verifier result for one action."""

    path = (
        Path(output_dir)
        / "model_roles"
        / f"host_advisory_attempt_{int(attempt):02d}.json"
    )
    payload = read_json(path)
    if not isinstance(payload, dict) or not payload:
        return
    payload["actual_physical_verifier_result"] = {
        "physically_dispatched": trace_has_physical_dispatch(trace),
        "success": bool(report.success),
        "failure_type": str(report.failure_type),
        "transition_outcome": transition_outcome_from_report(report).value,
        "checkpoint_outcome": report.evidence.get("checkpoint_outcome", {}),
    }
    write_json(path, payload)


_GenerationResult = TypeVar("_GenerationResult")

_RECOVERY_MUTATION_LEVELS = frozenset(
    {"parameter_tuning", "coordination_mutation", "strategy_mutation"}
)


def _all_recovery_levels_deterministically_infeasible(
    stall_events: list[dict[str, object]],
) -> bool:
    """Return true only after every model-owned recovery level proved empty.

    Repeatedly selecting one or two local empty scopes does not prove that the
    checkpoint action space is empty.  In particular, strategy mutation remains
    UNKNOWN/available until its own deterministic feasibility check rejects it.
    """

    infeasible_levels = {
        str(event.get("selected_mutation_level", "")).strip()
        for event in stall_events
        if event.get("cycle_outcome") == MUTATION_SCOPE_INFEASIBLE
    }
    return _RECOVERY_MUTATION_LEVELS.issubset(infeasible_levels)


def _run_checkpoint_generation_cycles(
    *,
    tracker: GenerationStallTracker,
    checkpoint_id: str,
    generate: Callable[[int], _GenerationResult],
    on_stall: Callable[[dict], None] | None = None,
    on_retry: Callable[[int, dict], None] | None = None,
) -> _GenerationResult:
    """Retry complete pre-dispatch pipelines before sealing SYSTEM_STALL."""

    generation_cycle = 0
    infeasible_scope_events: list[dict[str, object]] = []
    while True:
        generation_cycle += 1
        try:
            result = generate(generation_cycle)
        except Exception as exc:
            role_stall = _role_stall_from_exception(exc)
            if role_stall is None:
                raise
            stall_event = tracker.record(
                checkpoint_id=checkpoint_id,
                generation_cycle=generation_cycle,
                role=role_stall.role,
                artifact_id=role_stall.artifact_id,
                error=str(role_stall),
                response_count=role_stall.response_count,
                **_role_stall_record_fields(role_stall),
            )
            if on_stall is not None:
                on_stall(stall_event)
            if stall_event.get("cycle_outcome") == MUTATION_SCOPE_INFEASIBLE:
                infeasible_scope_events.append(stall_event)
                if _all_recovery_levels_deterministically_infeasible(
                    infeasible_scope_events
                ):
                    raise HostContractError(
                        "all available recovery mutation levels were "
                        "deterministically infeasible at checkpoint "
                        f"{checkpoint_id}"
                    ) from exc
            if stall_event["system_stall"]:
                raise SystemStallError(
                    "three complete generation cycles produced no executable child "
                    f"at checkpoint {checkpoint_id}"
                ) from exc
            if on_retry is not None:
                on_retry(generation_cycle + 1, stall_event)
            continue
        tracker.reset_after_successful_role_cycle()
        return result


def _load_policy(
    path: Path,
    *,
    allow_legacy_invalid_parameters: bool = False,
) -> PolicyScript:
    """Load a policy, optionally retaining invalid calls as failed evidence only."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"resume policy at {path} must be a JSON object")
    nested = payload.get("policy")
    if isinstance(nested, dict):
        payload = nested

    assignments = tuple(
        RobotAssignment(
            robot_id=str(item["robot_id"]),
            role=str(item.get("role", "")),
            objective=str(item.get("objective", "")),
            synchronization_group=str(item.get("synchronization_group", "")),
            safety_constraints=tuple(str(value) for value in item.get("safety_constraints", ())),
            phase_id=str(item.get("phase_id", "")),
            contact_intent=str(item.get("contact_intent", "unspecified")),
            coordination_mode=_persisted_coordination_mode(
                item.get("coordination_mode", "sync")
            ),
        )
        for item in payload.get("assignments", ())
        if isinstance(item, dict)
    )
    robot_scripts = tuple(
        RobotPolicyScript(
            robot_id=str(item["robot_id"]),
            script_id=str(item["script_id"]),
            code=str(item["code"]),
            synchronization_group=str(item["synchronization_group"]),
            phase_id=str(item.get("phase_id", "")),
            contact_intent=str(item.get("contact_intent", "unspecified")),
            coordination_mode=_persisted_coordination_mode(
                item.get("coordination_mode", "sync")
            ),
        )
        for item in payload.get("robot_scripts", ())
        if isinstance(item, dict)
    )
    if {item.robot_id for item in robot_scripts} != {"front", "back"}:
        raise ValueError(f"resume policy at {path} must contain front/back robot_scripts")
    if {item.robot_id for item in assignments} != {"front", "back"}:
        raise ValueError(f"resume policy at {path} must contain front/back assignments")
    for script in robot_scripts:
        validate_step_policy_code(
            script.code,
            validate_parameters=not allow_legacy_invalid_parameters,
        )
    assignments = _normalize_coordination_assignment_roles(assignments, robot_scripts)

    subtasks = tuple(
        Subtask(
            name=str(item["name"]),
            description=str(item.get("description", "")),
            skill_name=str(item.get("skill_name", "")),
            parameters=dict(item.get("parameters", {})),
            guard=str(item.get("guard", "")),
        )
        for item in payload.get("subtasks", ())
        if isinstance(item, dict)
    )
    front_code = next(item.code for item in robot_scripts if item.robot_id == "front")
    policy = PolicyScript(
        script_id=str(payload["script_id"]),
        attempt=int(payload["attempt"]),
        code=str(payload.get("code") or front_code),
        plan=tuple(str(item) for item in payload.get("plan", ())),
        subtasks=subtasks,
        assignments=assignments,
        robot_scripts=robot_scripts,
        parent_script_id=str(payload.get("parent_script_id", "")),
        rationale=str(payload.get("rationale", "")),
        verification_contract=dict(payload.get("verification_contract", {})),
    )
    if policy.attempt < 1:
        raise ValueError(f"resume policy attempt must be positive, got {policy.attempt}")
    return policy


def _load_evaluation_report(path: Path) -> EvaluationReport:
    if not path.exists():
        preliminary_path = path.with_name("preliminary_physics_evaluation.json")
        if preliminary_path.exists():
            path = preliminary_path
    payload = read_json(path)
    if not payload:
        raise ValueError(f"missing persisted evaluation report: {path}")
    metrics = dict(payload.get("metrics", {}))
    failure_type = str(payload.get("failure_type", "script_error"))
    evidence = dict(payload.get("evidence", {}))
    required_displacement = float(
        metrics.get("required_step_displacement_m", 0.01)
    )
    visual_payload = evidence.get("visual_verifier", {})
    if (
        failure_type == "manual_review_required"
        and isinstance(visual_payload, dict)
    ):
        visual_report = _visual_report_from_payload(visual_payload)
        route_regression = _route_regression_evidence(
            visual_report,
            action=str(payload.get("failed_subtask", "")),
            metrics=metrics,
        )
        if route_regression:
            evidence["persisted_failure_type"] = failure_type
            evidence["resume_reclassification"] = (
                "confirmed executed-motion route risk is a strategy failure, "
                "not a missing-evidence system failure"
            )
            evidence["route_direction_verification"] = route_regression
            failure_type = "route_regression"
    if (
        failure_type == "no_payload_progress"
        and float(metrics.get("payload_step_displacement_xy_m", 0.0))
        >= required_displacement
        and float(metrics.get("payload_target_progress_m", 0.0))
        <= -required_displacement
    ):
        evidence["persisted_failure_type"] = failure_type
        evidence["resume_reclassification"] = (
            "measurable world displacement increased active-goal distance"
        )
        failure_type = "goal_regression"
    report = EvaluationReport(
        success=bool(payload.get("success", False)),
        failure_type=failure_type,
        failed_subtask=str(payload.get("failed_subtask", "")),
        metrics=metrics,
        evidence=evidence,
        recommendations=tuple(str(item) for item in payload.get("recommendations", ())),
        visual_summary=str(payload.get("visual_summary", "")),
        failure_level=str(payload.get("failure_level", "none")),
        diagnosis=str(payload.get("diagnosis", "")),
        allowed_repairs=tuple(str(item) for item in payload.get("allowed_repairs", ())),
        forbidden_repairs=tuple(str(item) for item in payload.get("forbidden_repairs", ())),
    )
    trace_path = path.with_name("execution_trace.json")
    if not report.success and trace_path.exists():
        monitored_failure = _monitored_physical_failure(
            _load_execution_trace(trace_path)
        )
        if monitored_failure and monitored_failure != report.failure_type:
            report = replace(
                report,
                failure_type=monitored_failure,
                failure_level="none",
                diagnosis="",
                allowed_repairs=(),
                forbidden_repairs=(),
                evidence={
                    **report.evidence,
                    "persisted_failure_type": report.failure_type,
                    "resume_reclassification": (
                        "preserved the earliest online physical-safety abort from "
                        "execution_trace instead of its later free-body endpoint"
                    ),
                    "monitored_physical_failure": monitored_failure,
                },
            )
    return annotate_evaluation_report(report)


def _load_execution_trace(path: Path) -> ExecutionTrace:
    payload = read_json(path)
    if not payload:
        raise ValueError(f"missing persisted execution trace: {path}")
    skill_calls = tuple(
        SkillCallRecord(
            index=int(item.get("index", index)),
            skill_name=str(item.get("skill_name", "")),
            parameters=dict(item.get("parameters", {})),
            status=str(item.get("status", "")),
            started_at_unix=float(item.get("started_at_unix", 0.0)),
            finished_at_unix=float(item.get("finished_at_unix", 0.0)),
            artifact_path=str(item.get("artifact_path", "")),
            stdout=str(item.get("stdout", "")),
            stderr=str(item.get("stderr", "")),
            robot_ids=tuple(str(value) for value in item.get("robot_ids", ())),
            effective_parameters=dict(item.get("effective_parameters", {})),
        )
        for index, item in enumerate(payload.get("skill_calls", ()), start=1)
        if isinstance(item, dict)
    )
    robot_executions = []
    for item in payload.get("robot_executions", ()):
        if not isinstance(item, dict):
            continue
        planned_calls = tuple(
            PlannedSkillCall(
                index=int(call.get("index", index)),
                method=str(call.get("method", "")),
                parameters=dict(call.get("parameters", {})),
            )
            for index, call in enumerate(item.get("planned_calls", ()), start=1)
            if isinstance(call, dict)
        )
        robot_executions.append(
            RobotExecutionTrace(
                robot_id=str(item.get("robot_id", "")),
                script_id=str(item.get("script_id", "")),
                status=str(item.get("status", "")),
                synchronization_group=str(item.get("synchronization_group", "")),
                planned_calls=planned_calls,
                error=str(item.get("error", "")),
                phase_id=str(item.get("phase_id", "")),
                contact_intent=str(item.get("contact_intent", "unspecified")),
                coordination_mode=_persisted_coordination_mode(
                    item.get("coordination_mode", "sync")
                ),
            )
        )
    return ExecutionTrace(
        script_id=str(payload.get("script_id", "")),
        attempt=int(payload.get("attempt", 0)),
        status=str(payload.get("status", "")),
        skill_calls=skill_calls,
        robot_executions=tuple(robot_executions),
        error=str(payload.get("error", "")),
        artifact_dir=str(payload.get("artifact_dir", "")),
        verification_contract=dict(payload.get("verification_contract", {})),
    )


def _persisted_coordination_mode(value: object) -> str:
    mode = str(value).strip().lower() or "sync"
    if mode not in {"sync", "async"}:
        raise ValueError(
            f"persisted coordination_mode must be sync or async, got {value!r}"
        )
    return mode


def append_rollout_evidence(
    path: Path,
    spec: str,
    *,
    dry_run: bool,
    policy: PolicyScript | None = None,
    report: EvaluationReport | None = None,
    trace: ExecutionTrace | None = None,
    checkpoint_before: str = "",
    checkpoint_after: str = "",
    controller_decision: dict[str, object] | None = None,
    strategy_store: StrategyMemoryStore | None = None,
    strategy_phase: str = "any",
    role_pipeline: ModelRolePipeline | None = None,
    physically_dispatched: bool | None = None,
    physical_transaction_completed: bool | None = None,
) -> dict[str, object]:
    """Durably record one rollout, then run optional memory abstraction.

    The first JSONL event is authoritative raw physical evidence and is flushed
    before any provider call. The second event only reports the best-effort
    Memory Update result. This keeps provider outages out of the physics loop
    and lets offline rebuild recover every completed rollout.
    """

    if not spec and (policy is None or report is None):
        return {}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "physical_rollout_finalized",
        "dry_run": dry_run,
        "model_spec_candidate": spec,
    }
    if controller_decision:
        payload["evolution_controller"] = dict(controller_decision)
    if policy is not None:
        payload["script_id"] = policy.script_id
        payload["attempt"] = policy.attempt
        payload["executed_skill"] = _policy_action_method(
            policy,
            validate_parameters=False,
        )
        payload["coordination_mode"] = _policy_coordination_mode(policy)
        payload["coordination"] = [
            {
                "robot_id": script.robot_id,
                "phase_id": script.phase_id,
                "contact_intent": script.contact_intent,
                "synchronization_group": script.synchronization_group,
                "coordination_mode": script.coordination_mode,
            }
            for script in policy.robot_scripts
        ]
        payload["policy_execution_parameters"] = _policy_execution_parameters(policy)
    payload["strategy_context"] = {"phase": strategy_phase}
    if trace is not None:
        payload["backend_execution_parameters"] = [
            {
                "skill": call.skill_name,
                "robot_ids": list(call.robot_ids),
                "parameters": call.effective_parameters or call.parameters,
            }
            for call in trace.skill_calls
            if call.skill_name != "stop_all"
        ]
    if physically_dispatched is None:
        physically_dispatched = bool(
            not dry_run and trace_has_physical_dispatch(trace)
        )
    else:
        physically_dispatched = bool(physically_dispatched and not dry_run)
    if physical_transaction_completed is None:
        physical_transaction_completed = bool(
            physically_dispatched
            and (
                trace_physical_transaction_completed(trace)
                or (trace is not None and _monitored_physical_failure(trace) is not None)
            )
        )
    else:
        physical_transaction_completed = bool(
            physically_dispatched and physical_transaction_completed
        )
    if report is not None and isinstance(report.evidence, dict):
        terminal = report.evidence.get("post_execution_model_terminal", {})
        if isinstance(terminal, dict) and terminal:
            physical_transaction_completed = False
    payload["physically_dispatched"] = physically_dispatched
    payload["physical_transaction_completed"] = physical_transaction_completed
    if report is not None:
        failure_classification = classify_report(report)
        payload["verifier"] = {
            "step_accepted": bool(report.success),
            "failure_type": report.failure_type,
            "failure_level": failure_classification.level,
            "diagnosis": failure_classification.diagnosis,
            "allowed_repairs": list(failure_classification.allowed_repairs),
            "forbidden_repairs": list(failure_classification.forbidden_repairs),
            "failed_subtask": report.failed_subtask,
            "task_complete": bool(report.metrics.get("task_complete", 0.0)),
            "checkpoint_before": checkpoint_before,
            "checkpoint_after": checkpoint_after,
            "recommendations": list(report.recommendations),
            "metrics": dict(report.metrics),
        }
        transition = report.evidence.get("transition_decision", {})
        if isinstance(transition, dict) and transition:
            final_transition_outcome = transition_outcome_from_report(report)
            payload["verifier"]["transition_outcome"] = (
                final_transition_outcome.value
            )
            payload["verifier"]["positive_route_memory_allowed"] = bool(
                report.success
                and final_transition_outcome.value
                in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
            )
        accepted_route_quality = report.evidence.get("accepted_route_quality")
        if isinstance(accepted_route_quality, dict) and accepted_route_quality:
            payload["verifier"]["accepted_route_quality"] = dict(
                accepted_route_quality
            )
        simulation_observation = report.evidence.get("simulation_observation")
        if isinstance(simulation_observation, dict):
            payload["simulation_observation"] = simulation_observation
        visual_observation = _rollout_visual_observation(report)
        if visual_observation:
            payload["verifier"]["visual_observation"] = visual_observation
    rollout_id = _rollout_event_id(payload)
    payload["rollout_id"] = rollout_id
    _append_jsonl_event(path, payload)

    model_memory_update = None
    memory_event: dict[str, object] = {
        "kind": "memory_update_result",
        "rollout_id": rollout_id,
        "script_id": str(payload.get("script_id", "")),
        "attempt": payload.get("attempt"),
        "status": "not_requested",
        "memory_changed": False,
    }
    if (
        role_pipeline is not None
        and strategy_store is not None
        and not dry_run
        and policy is not None
        and report is not None
        and physical_transaction_completed
    ):
        try:
            role_decisions = role_pipeline.responses_for_artifact(
                f"attempt_{policy.attempt:02d}"
            )
            payload["model_role_decisions"] = {
                role: role_decisions[role]
                for role in (
                    "task_orchestrator",
                    "evolution_manager",
                    "mutation_operator",
                )
                if role in role_decisions
            }
            model_memory_update = role_pipeline.propose_memory_update(
                {
                    "rollout": payload,
                    "promotion_contract": {
                        "physics_verifier_is_authoritative": True,
                        "store_behavior_abstraction_only": True,
                        "forbidden_content": [
                            "action magnitudes",
                            "checkpoint poses",
                            "scene-specific routes",
                            "unverified causal claims",
                        ],
                    },
                },
                artifact_id=f"attempt_{policy.attempt:02d}",
            )
            memory_event["status"] = "model_update_ready"
            memory_event["model_memory_update"] = model_memory_update
        except Exception as exc:  # noqa: BLE001 - memory is never a run-fatal service
            model_memory_update = {
                "promote": False,
                "source": "memory_update_unavailable",
            }
            memory_event["status"] = "memory_update_unavailable"
            memory_event["memory_update_error"] = {
                "promoted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "policy": (
                    "Raw rollout evidence is already durable; keep active memory "
                    "unchanged and continue the experiment."
                ),
            }
    memory_changed = False
    if strategy_store is not None:
        try:
            memory_changed = strategy_store.update_from_rollout(
                payload,
                default_phase=strategy_phase,
                model_proposal=model_memory_update,
            )
        except Exception as exc:  # noqa: BLE001 - invalid memory cannot block execution
            memory_event["status"] = "memory_promotion_unavailable"
            memory_event["memory_promotion_error"] = {
                "promoted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "policy": "retain raw evidence and keep active memory unchanged",
            }
    if strategy_store is not None and memory_changed:
        strategy_store.save()
        memory_event["strategy_memory_changed"] = True
        if memory_event["status"] not in {
            "memory_update_unavailable",
            "memory_promotion_unavailable",
        }:
            memory_event["status"] = "promoted"
    elif memory_event["status"] == "model_update_ready":
        memory_event["status"] = "not_promoted"
    memory_event["memory_changed"] = memory_changed
    if role_pipeline is not None or strategy_store is not None:
        _append_jsonl_event(path, memory_event)
    payload["memory_update"] = memory_event
    return payload


def _append_jsonl_event(path: Path, event: dict[str, object]) -> None:
    """Append and fsync one event before dependent work begins."""

    event_kind = str(event.get("kind", "")).strip()
    rollout_id = str(event.get("rollout_id", "")).strip()
    if rollout_id and event_kind in {
        "physical_rollout_finalized",
        "memory_update_result",
    } and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                existing = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"cannot append to corrupted rollout ledger {path}"
                ) from exc
            if not isinstance(existing, dict):
                continue
            if (
                str(existing.get("kind", "")).strip() == event_kind
                and str(existing.get("rollout_id", "")).strip() == rollout_id
            ):
                if existing == event:
                    return
                raise RuntimeError(
                    "conflicting duplicate rollout event "
                    f"kind={event_kind} rollout_id={rollout_id} in {path}"
                )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _runner_json_default(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_runner_json(path: Path, payload: object) -> None:
    """Atomically replace a small runner-owned recovery artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=_runner_json_default,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_runner_jsonl(path: Path, payloads: list[object]) -> None:
    """Atomically replace the derived evolution-event projection."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    default=_runner_json_default,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _begin_physical_transaction_commit(
    *,
    attempt_dir: Path,
    attempt: int,
    script_id: str,
    checkpoint_before: str,
    checkpoint_after: str,
    checkpoint_committed: bool,
    memory_required: bool,
    outcome_finalized: bool = True,
) -> Path:
    """Durably mark a confirmed dispatch before outcome-dependent side effects."""

    path = Path(attempt_dir) / "physical_transaction_commit.json"
    payload: dict[str, object] = {
        "schema_version": FORMAL_TRANSACTION_JOURNAL_SCHEMA_VERSION,
        "attempt": int(attempt),
        "script_id": str(script_id),
        "checkpoint_before": str(checkpoint_before),
        "checkpoint_after": str(checkpoint_after),
        "checkpoint_committed": bool(checkpoint_committed),
        "outcome_finalized": bool(outcome_finalized),
        "memory_required": bool(memory_required),
        "status": "pending",
        "stages": dict.fromkeys(FORMAL_TRANSACTION_JOURNAL_STAGES, False) | {
            "dispatch_confirmed": True,
            "outcome": bool(outcome_finalized),
            "checkpoint_lineage": bool(
                outcome_finalized and not checkpoint_committed
            ),
            "memory_update": not memory_required,
        },
        "stage_details": {},
    }
    _atomic_write_runner_json(path, payload)
    return path


def _finalize_physical_transaction_outcome(
    path: Path | None,
    *,
    checkpoint_after: str,
    checkpoint_committed: bool,
    step_accepted: bool,
) -> None:
    """Bind one confirmed dispatch to its canonical restored/committed state."""

    if path is None:
        return
    payload = read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"physical transaction journal is missing: {path}")
    canonical_after = str(checkpoint_after).strip()
    if not canonical_after:
        raise RuntimeError("finalized physical transaction has no checkpoint_after")
    existing_after = str(payload.get("checkpoint_after", "")).strip()
    if bool(payload.get("outcome_finalized", False)) and existing_after not in {
        "",
        canonical_after,
    }:
        raise RuntimeError(
            "physical transaction outcome changed checkpoint identity: "
            f"{existing_after} != {canonical_after}"
        )
    payload["checkpoint_after"] = canonical_after
    payload["checkpoint_committed"] = bool(checkpoint_committed)
    payload["step_accepted"] = bool(step_accepted)
    payload["outcome_finalized"] = True
    stages = payload.get("stages", {})
    if not isinstance(stages, dict) or "outcome" not in stages:
        raise RuntimeError("physical transaction journal has no outcome stage")
    stages["outcome"] = True
    if not checkpoint_committed:
        stages["checkpoint_lineage"] = True
    payload["stages"] = stages
    stage_details = payload.get("stage_details", {})
    stage_details = stage_details if isinstance(stage_details, dict) else {}
    stage_details["outcome"] = {
        "checkpoint_after": canonical_after,
        "checkpoint_committed": bool(checkpoint_committed),
        "step_accepted": bool(step_accepted),
    }
    if not checkpoint_committed:
        stage_details["checkpoint_lineage"] = {
            "restored_checkpoint": canonical_after,
            "new_checkpoint_committed": False,
        }
    payload["stage_details"] = stage_details
    _atomic_write_runner_json(Path(path), payload)


def _advance_physical_transaction_commit(
    path: Path | None,
    stage: str,
    *,
    details: dict[str, object] | None = None,
) -> None:
    """Mark one idempotent transaction stage durable for interruption recovery."""

    if path is None:
        return
    payload = read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"physical transaction journal is missing: {path}")
    stages = payload.get("stages", {})
    if not isinstance(stages, dict) or stage not in stages:
        raise RuntimeError(f"unknown physical transaction stage {stage!r}")
    stages[stage] = True
    payload["stages"] = stages
    stage_details = payload.get("stage_details", {})
    stage_details = stage_details if isinstance(stage_details, dict) else {}
    if details:
        if stage == "memory_update":
            raw_details = stage_details.get("raw_rollout", {})
            raw_details = raw_details if isinstance(raw_details, dict) else {}
            raw_rollout_id = str(raw_details.get("rollout_id", "")).strip()
            memory_rollout_id = str(details.get("rollout_id", "")).strip()
            if (
                raw_rollout_id
                and memory_rollout_id
                and raw_rollout_id != memory_rollout_id
            ):
                raise RuntimeError(
                    "raw rollout and memory update IDs disagree: "
                    f"{raw_rollout_id} != {memory_rollout_id}"
                )
        stage_details[stage] = dict(details)
    payload["stage_details"] = stage_details
    nonterminal = tuple(key for key in stages if key != "terminal_metrics")
    artifacts_committed = all(bool(stages[key]) for key in nonterminal)
    payload["status"] = (
        "sealed"
        if artifacts_committed and bool(stages.get("terminal_metrics"))
        else "artifacts_committed"
        if artifacts_committed
        else "pending"
    )
    _atomic_write_runner_json(Path(path), payload)


def _mark_transaction_journals_terminal_metrics(
    output_dir: Path,
    *,
    require_manifest: bool = False,
) -> None:
    """Link sealed terminal metrics to every completed physical transaction."""

    output_dir = Path(output_dir)
    journals = sorted(
        output_dir.glob("attempt_*/physical_transaction_commit.json")
    )
    if not journals:
        return
    required = [output_dir / "metrics.json"]
    if require_manifest:
        required.append(output_dir / "run_manifest.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "cannot seal physical transactions before terminal artifacts exist: "
            + ", ".join(missing)
        )
    for path in journals:
        details = {"metrics": "metrics.json"}
        if (output_dir / "run_manifest.json").is_file():
            details["manifest"] = "run_manifest.json"
        _advance_physical_transaction_commit(
            path,
            "terminal_metrics",
            details=details,
        )


def _controller_metrics_with_dispatch_journals(
    controller_metrics: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    """Reconcile terminal physical count with pre-evaluation dispatch markers."""

    reconciled = dict(controller_metrics)
    journals = []
    for path in sorted(Path(output_dir).glob("attempt_*/physical_transaction_commit.json")):
        journal = read_json(path)
        stages = journal.get("stages", {}) if isinstance(journal, dict) else {}
        if isinstance(stages, dict) and stages.get("dispatch_confirmed") is True:
            journals.append(journal)
    journal_count = len(journals)
    try:
        controller_count = int(reconciled.get("physical_iterations", 0) or 0)
    except (TypeError, ValueError):
        controller_count = 0
    physical_count = max(controller_count, journal_count)
    pending = max(0, journal_count - controller_count)
    reconciled["physical_iterations"] = physical_count
    reconciled["journal_confirmed_physical_iterations"] = journal_count
    reconciled["pending_physical_evaluation_iterations"] = pending
    reconciled["physical_dispatch_accounting_complete"] = bool(
        controller_count == journal_count
    )
    budget = reconciled.get("budget", {})
    budget = dict(budget) if isinstance(budget, dict) else {}
    maximum = budget.get("maximum_physical_iterations")
    unlimited = bool(budget.get("unlimited", maximum is None))
    budget["used"] = physical_count
    if not unlimited and maximum is not None:
        try:
            maximum_count = max(0, int(maximum))
        except (TypeError, ValueError):
            maximum_count = 0
        budget["remaining"] = max(0, maximum_count - physical_count)
        budget["exhausted"] = bool(
            budget.get("exhausted", False)
            or (
                physical_count >= maximum_count
                and not bool(reconciled.get("task_success", False))
            )
        )
    reconciled["budget"] = budget
    return reconciled


def _rollout_event_id(payload: dict[str, object]) -> str:
    identity = {
        "script_id": payload.get("script_id", ""),
        "attempt": payload.get("attempt"),
        "checkpoint_before": payload.get("verifier", {}).get("checkpoint_before", "")
        if isinstance(payload.get("verifier"), dict)
        else "",
        "checkpoint_after": payload.get("verifier", {}).get("checkpoint_after", "")
        if isinstance(payload.get("verifier"), dict)
        else "",
    }
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]


def _rollout_visual_observation(report: EvaluationReport) -> dict[str, object]:
    """Keep only the visual safety verdict in dynamic rollout records."""

    visual = report.evidence.get("visual_verifier", {})
    if not isinstance(visual, dict) or visual.get("status") != "ok":
        return {}
    evidence = visual.get("evidence", {})
    evidence = evidence if isinstance(evidence, dict) else {}
    observation = {
        "source": "qwen_vl_external_camera",
        "model_name": str(visual.get("model_name", "")),
        "confidence": visual.get("confidence"),
        "summary": str(visual.get("summary", report.visual_summary)).strip()[:600],
        "scene_visible": evidence.get("scene_visible"),
        "payload_on_floor": evidence.get("payload_on_floor"),
        "robots_standing_stably": evidence.get("robots_standing_stably"),
        "wall_collision_or_stuck": evidence.get("wall_collision_or_stuck"),
        "payload_placed_on_target": evidence.get("payload_placed_on_target"),
        "failure_analysis": evidence.get("failure_analysis", {}),
        "visible_evidence": [
            str(item)[:240] for item in evidence.get("visible_evidence", ())
        ][:4],
    }
    return {
        key: value
        for key, value in observation.items()
        if value not in (None, "", [])
    }


def _policy_execution_parameters(policy: PolicyScript) -> list[dict[str, object]]:
    """Extract literal Agent SDK calls for verifier-grounded memory."""

    scripts = list(policy.robot_scripts)
    if not scripts:
        scripts = [
            RobotPolicyScript(
                robot_id=robot_id,
                script_id=f"{robot_id}_{policy.script_id}",
                code=policy.code,
                synchronization_group="legacy_team",
            )
            for robot_id in ("front", "back")
        ]
    extracted: list[dict[str, object]] = []
    for script in scripts:
        try:
            tree = ast.parse(script.code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Attribute)
                or not isinstance(node.func.value, ast.Name)
                or node.func.value.id != "api"
                or node.func.attr == "stop_all"
            ):
                continue
            parameters: dict[str, object] = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                try:
                    parameters[keyword.arg] = ast.literal_eval(keyword.value)
                except (TypeError, ValueError):
                    parameters[keyword.arg] = "<non_literal>"
            extracted.append(
                {
                    "robot_id": script.robot_id,
                    "skill": node.func.attr,
                    "parameters": parameters,
                }
            )
            break
    return extracted


def _validate_live_pivot_roles(
    policy: PolicyScript,
    runtime_state: dict,
) -> None:
    """Reject a pivot whose temporary roles contradict current geometry."""

    calls = [
        item
        for item in _policy_execution_parameters(policy)
        if item.get("skill") == COORDINATED_PIVOT
    ]
    if not calls:
        return
    directions = {
        str(item.get("parameters", {}).get("turn_direction", "")).strip()
        for item in calls
        if isinstance(item.get("parameters"), dict)
    }
    if len(directions) != 1:
        raise TurnEpisodeError(
            "coordinated_pivot preflight requires one shared turn_direction"
        )
    direction = next(iter(directions))
    expected = pivot_role_assignment(runtime_state, direction)
    if set(expected) != {"front", "back"}:
        raise TurnEpisodeError(
            "coordinated_pivot preflight cannot derive inner/outer roles from "
            "the current front/back poses"
        )
    actual = {
        str(item.get("robot_id", "")).strip(): str(
            item.get("parameters", {}).get("role", "")
        ).strip()
        for item in calls
        if isinstance(item.get("parameters"), dict)
    }
    if actual != expected:
        raise TurnEpisodeError(
            "coordinated_pivot live role mismatch: expected "
            f"{expected} for turn_direction={direction}, got {actual}; "
            "front/back are immutable robot IDs"
        )


def _validate_pivot_candidate_readiness(
    policy: PolicyScript,
    runtime_state: dict,
    *,
    hard_collision_margin_m: float | None = None,
) -> dict[str, object]:
    """Check one exact pivot candidate without selecting its replacement."""

    calls = [
        item
        for item in _policy_execution_parameters(policy)
        if item.get("skill") == COORDINATED_PIVOT
    ]
    if not calls:
        return {}
    directions = {
        str(item.get("parameters", {}).get("turn_direction", "")).strip().lower()
        for item in calls
        if isinstance(item.get("parameters"), dict)
    }
    angles = {
        item.get("parameters", {}).get("angle_deg")
        for item in calls
        if isinstance(item.get("parameters"), dict)
    }
    if len(directions) != 1 or len(angles) != 1:
        raise TurnEpisodeError(
            "coordinated_pivot readiness requires one shared literal direction and angle"
        )
    direction = next(iter(directions))
    try:
        angle_deg = abs(float(next(iter(angles))))
    except (TypeError, ValueError) as exc:
        raise TurnEpisodeError(
            "coordinated_pivot readiness requires a literal numeric angle_deg"
        ) from exc
    roles = {
        str(item.get("robot_id", "")).strip(): str(
            item.get("parameters", {}).get("role", "")
        ).strip()
        for item in calls
        if isinstance(item.get("parameters"), dict)
    }
    clearance = pivot_candidate_swept_clearance(
        runtime_state,
        turn_direction=direction,
        angle_deg=angle_deg,
        roles=roles,
        hard_collision_margin_m=hard_collision_margin_m,
    )
    available = bool(clearance.get("available", False))
    evidence: dict[str, object] = {
        "schema_version": 1,
        "kind": "candidate_specific_full_team_pivot_readiness",
        "decision": (
            "ready"
            if available and clearance.get("executable", False)
            else "not_ready"
            if available
            else "unavailable"
        ),
        "physically_dispatched": False,
        "candidate": {
            "primitive": COORDINATED_PIVOT,
            "turn_direction": direction,
            "angle_deg": angle_deg,
            "coordination_mode": _policy_coordination_mode(policy),
            "roles": roles,
        },
        "swept_clearance": clearance,
        "decision_owner": "host_geometry_verifier",
        "replacement_action_owner": "model_roles",
    }
    if available and not bool(clearance.get("executable", False)):
        raise TurnReadinessError(
            "PREFLIGHT_TURN_CLEARANCE_INSUFFICIENT: this exact model-authored pivot "
            "lacks complete-team swept clearance at the current trusted checkpoint",
            evidence=evidence,
        )
    return evidence


def _validate_policy_turn_preflight(
    policy: PolicyScript,
    *,
    runtime_state: dict,
    turn_episode: dict,
    formal_v50_lite: bool = False,
) -> dict[str, object]:
    """Apply turn continuity and live role geometry before every dispatch."""

    if formal_v50_lite:
        # Route, heading, role mapping, and turn-episode recommendations remain
        # model-owned in V50-lite. Only an explicit overlap/contact prediction
        # for the exact executable pivot is a hard pre-dispatch rejection.
        return _validate_pivot_candidate_readiness(
            policy,
            runtime_state,
            hard_collision_margin_m=0.0,
        )

    local_navigation = (
        policy.verification_contract.get("local_navigation", {})
        if isinstance(policy.verification_contract, dict)
        else {}
    )
    route_clearance_conflict = (
        policy.verification_contract.get("route_clearance_conflict", {})
        if isinstance(policy.verification_contract, dict)
        else {}
    )
    turn_clearance_replan = (
        policy.verification_contract.get("turn_clearance_replan", {})
        if isinstance(policy.verification_contract, dict)
        else {}
    )
    turn_clearance_replan = (
        turn_clearance_replan if isinstance(turn_clearance_replan, dict) else {}
    )
    route_recovery = (
        policy.verification_contract.get("route_recovery", {})
        if isinstance(policy.verification_contract, dict)
        else {}
    )
    route_recovery = route_recovery if isinstance(route_recovery, dict) else {}
    if isinstance(route_clearance_conflict, dict) and route_clearance_conflict.get(
        "active", False
    ):
        checkpoint = runtime_state.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        expected_scope = str(
            route_clearance_conflict.get("scope_checkpoint_id", "")
        ).strip()
        live_scope = str(checkpoint.get("checkpoint_id", "")).strip()
        if expected_scope and live_scope != expected_scope:
            raise TurnEpisodeError(
                "route-clearance repair contract is stale for the live checkpoint: "
                f"expected {expected_scope!r}, got {live_scope!r}"
            )
    policy_method, policy_objective = _policy_motion_objective(policy)
    policy_mode = _policy_coordination_mode(policy)
    turn_clearance_allowed = bool(
        turn_clearance_replan.get("active", False)
        and policy_method
        == turn_clearance_replan.get(
            "required_primitive", COORDINATED_LINEAR_TRANSPORT
        )
        and policy_mode
        == turn_clearance_replan.get("required_coordination_mode", "sync")
        and policy_objective
        in set(turn_clearance_replan.get("allowed_translation_directions", ()))
    )
    if turn_clearance_replan.get("active", False):
        checkpoint = runtime_state.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        expected_scope = str(
            turn_clearance_replan.get("scope_checkpoint_id", "")
        ).strip()
        live_scope = str(checkpoint.get("checkpoint_id", "")).strip()
        if expected_scope and live_scope != expected_scope:
            raise TurnEpisodeError(
                "turn-clearance replan contract is stale for the live checkpoint: "
                f"expected {expected_scope!r}, got {live_scope!r}"
            )
        if not turn_clearance_allowed:
            raise TurnEpisodeError(
                "turn-clearance replan permits exactly one sync "
                "coordinated_linear_transport in "
                f"{turn_clearance_replan.get('allowed_translation_directions', [])}"
            )
    if route_recovery.get("active", False):
        checkpoint = runtime_state.get("checkpoint", {})
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        expected_scope = str(
            route_recovery.get("scope_checkpoint_id", "")
        ).strip()
        live_scope = str(checkpoint.get("checkpoint_id", "")).strip()
        if expected_scope and live_scope != expected_scope:
            raise TurnEpisodeError(
                "route-recovery contract is stale for the live checkpoint: "
                f"expected {expected_scope!r}, got {live_scope!r}"
            )
    try:
        validate_strategy_against_local_navigation(
            method=policy_method,
            objective=policy_objective,
            contract=local_navigation,
            allow_route_recovery=bool(route_recovery.get("active", False)),
            allow_turn_clearance_replan=turn_clearance_allowed,
            route_clearance_conflict_contract=route_clearance_conflict,
        )
    except ValueError as exc:
        raise TurnEpisodeError(str(exc)) from exc
    final_target_turn = str(
        local_navigation.get("required_turn_direction", "")
    ).strip()
    validate_policy_for_turn_episode(
        policy,
        turn_episode,
        allow_final_target_reorientation=bool(
            policy_method == COORDINATED_PIVOT
            and policy_objective == final_target_turn
        ),
        allow_route_clearance_repair=bool(
            turn_clearance_allowed
            or isinstance(route_clearance_conflict, dict)
            and route_clearance_conflict.get("active", False)
            and (
                policy_method == COORDINATED_PIVOT
                and policy_objective
                in set(
                    route_clearance_conflict.get(
                        "safe_heading_correction_turn_directions",
                        (),
                    )
                )
                or policy_method == COORDINATED_LINEAR_TRANSPORT
                and policy_objective
                in set(
                    route_clearance_conflict.get(
                        "route_reentry_translation_directions",
                        (),
                    )
                )
            )
        ),
    )
    _validate_live_pivot_roles(policy, runtime_state)
    return _validate_pivot_candidate_readiness(policy, runtime_state)


def _archive_accepted_action_visuals(
    *,
    output_dir: Path,
    attempt: int,
    checkpoint_id: str,
    policy: PolicyScript,
    post_visual: VisualObservation,
    report: EvaluationReport,
) -> dict[str, object]:
    """Copy only verifier-accepted post-action frames into one review directory."""

    archive_dir = output_dir / "accepted_step_visuals"
    archive_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual_observations"
    sources = {
        path.resolve(strict=False)
        for path in visual_dir.glob(f"attempt_{attempt:02d}_post*.png")
        if path.is_file()
    }
    frame_path = Path(post_visual.frame_path)
    if frame_path.is_file():
        sources.add(frame_path.resolve(strict=False))

    copied: list[dict[str, str]] = []
    prefix = f"{checkpoint_id}_attempt_{attempt:02d}"
    for source in sorted(sources):
        marker = f"attempt_{attempt:02d}_post"
        label = source.stem
        if marker in source.stem:
            label = source.stem.split(marker, 1)[1].strip("_") or "global"
        destination = archive_dir / f"{prefix}_{label}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        copied.append(
            {
                "kind": label,
                "source": str(source),
                "archived": str(destination),
            }
        )

    entry: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "attempt": attempt,
        "script_id": policy.script_id,
        "primitive": _policy_action_method(policy),
        "visual_summary": report.visual_summary,
        "frames": copied,
    }
    write_json(archive_dir / f"{prefix}.json", entry)
    index_path = archive_dir / "index.json"
    current = read_json(index_path)
    entries = [
        item
        for item in current.get("accepted_actions", ())
        if isinstance(item, dict)
        and not (
            item.get("checkpoint_id") == checkpoint_id
            and item.get("attempt") == attempt
        )
    ]
    entries.append(entry)
    entries.sort(
        key=lambda item: (
            _accepted_checkpoint_index(str(item.get("checkpoint_id", ""))),
            int(item.get("attempt", 0)),
        )
    )
    write_json(
        index_path,
        {
            "kind": "accepted_action_visual_index",
            "accepted_actions": entries,
        },
    )
    print(
        "[cap-box-carry] archived accepted post-action visuals "
        f"checkpoint={checkpoint_id} frames={len(copied)} dir={archive_dir}",
        flush=True,
    )
    return entry


def _print_team_assignment(policy) -> None:
    print(
        f"[cap-box-carry] team_plan script={policy.script_id} attempt={policy.attempt}",
        flush=True,
    )
    for assignment in policy.assignments:
        print(
            f"[cap-box-carry] assign robot={assignment.robot_id} role={assignment.role} "
            f"phase={assignment.phase_id or '<inferred>'} "
            f"contact_intent={assignment.contact_intent} "
            f"sync_group={assignment.synchronization_group} "
            f"coordination_mode={assignment.coordination_mode} objective={assignment.objective}",
            flush=True,
        )
    for script in policy.robot_scripts:
        print(
            f"[cap-box-carry] robot_script robot={script.robot_id} path="
            f"attempt_{policy.attempt:02d}/{script.script_id}.py "
            f"phase={script.phase_id or '<inferred>'} contact_intent={script.contact_intent} "
            f"sync_group={script.synchronization_group} "
            f"coordination_mode={script.coordination_mode}",
            flush=True,
        )
    for call in _policy_execution_parameters(policy):
        print(
            f"[cap-box-carry] executable robot={call['robot_id']} "
            f"skill={call['skill']} parameters={call['parameters']}",
            flush=True,
        )


def _persist_task_distribution(
    output_dir: Path,
    policy,
    *,
    attempt_dir: Path | None = None,
    physical_dispatch_status: str = "not_started",
    robot_ports: dict[str, int] | None = None,
    trace=None,
    report=None,
    checkpoint_before: str = "",
    checkpoint_after: str = "",
    rollback_performed: bool = False,
) -> None:
    """Expose the model's team decision before any physical command is dispatched."""

    target_dir = attempt_dir or output_dir / f"attempt_{policy.attempt:02d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    script_paths: dict[str, str] = {}
    for script in policy.robot_scripts:
        script_path = target_dir / f"{script.script_id}.py"
        script_path.write_text(script.code, encoding="utf-8")
        script_paths[script.robot_id] = str(script_path)
    execution_by_robot = {
        item.robot_id: item for item in getattr(trace, "robot_executions", ())
    }
    assignment_by_robot = {item.robot_id: item for item in policy.assignments}
    robot_dispatch = {}
    for script in policy.robot_scripts:
        execution = execution_by_robot.get(script.robot_id)
        assignment = assignment_by_robot.get(script.robot_id)
        robot_dispatch[script.robot_id] = {
            "role": "" if assignment is None else assignment.role,
            "objective": "" if assignment is None else assignment.objective,
            "synchronization_group": script.synchronization_group,
            "coordination_mode": script.coordination_mode,
            "phase_id": script.phase_id,
            "contact_intent": script.contact_intent,
            "script_path": script_paths[script.robot_id],
            "sonic_zmq_port": None if robot_ports is None else robot_ports.get(script.robot_id),
            "script_status": "not_started" if execution is None else execution.status,
            "planned_calls": (
                []
                if execution is None
                else [asdict(call) for call in execution.planned_calls]
            ),
        }
    distribution = {
        "script_id": policy.script_id,
        "attempt": policy.attempt,
        "rationale": policy.rationale,
        "plan": list(policy.plan),
        "assignments": [asdict(item) for item in policy.assignments],
        "robot_script_paths": script_paths,
        "dispatch_mode": "atomic_team_publish_sync_or_async",
        "coordination_mode": _policy_coordination_mode(policy),
        "physical_dispatch_status": physical_dispatch_status,
        "robot_dispatch": robot_dispatch,
        "physical_verification": (
            {}
            if report is None
            else {
                "step_accepted": bool(report.success),
                "failure_type": report.failure_type,
                "failed_subtask": report.failed_subtask,
                "metrics": report.metrics,
            }
        ),
        "checkpoint": {
            "before": checkpoint_before,
            "after": checkpoint_after,
            "rollback_performed": bool(rollback_performed),
        },
    }
    write_json(target_dir / "team_plan.json", policy)
    write_json(target_dir / "task_distribution.json", distribution)
    write_json(output_dir / "latest_task_distribution.json", distribution)


def _initial_visual_cache_key(
    *,
    task: BoxCarryTask,
    runtime_state: dict,
    frame_path: str,
) -> str:
    """Identify one restored physical state and materially identical camera view."""

    if not frame_path:
        return ""
    payload = runtime_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    contacts = payload.get("contacts", {})
    contacts = contacts if isinstance(contacts, dict) else {}
    checkpoint = runtime_state.get("checkpoint", {})
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    robots = runtime_state.get("robots", {})
    robots = robots if isinstance(robots, dict) else {}

    def rounded(value: object) -> float | None:
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    state_signature = {
        "cache_schema": "initial_external_camera_v5_focused_posture_recheck",
        "task_stage": task.task_stage,
        "target": task.target,
        "checkpoint_id": checkpoint.get("checkpoint_id", ""),
        "payload": {
            key: rounded(payload.get(key))
            for key in ("x", "y", "z", "yaw_deg", "tilt_deg")
        },
        "payload_active": bool(payload.get("active", False)),
        "support_robots": sorted(str(item) for item in contacts.get("support_robots", ())),
        "support_geometries": sorted(
            str(item) for item in contacts.get("support_geometries", ())
        ),
        "floor_contact": bool(contacts.get("floor", False)),
        "wall_contacts": sorted(str(item) for item in contacts.get("walls", ())),
        "robots": {
            robot_id: {
                key: rounded(robot.get(key))
                for key in ("x", "y", "yaw_deg", "pelvis_height_m")
            }
            for robot_id in ("front", "back")
            if isinstance((robot := robots.get(robot_id)), dict)
        },
        "frame_perceptual_hash": _frame_perceptual_hash(Path(frame_path)),
    }
    encoded = json.dumps(
        state_signature,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_perceptual_hash(path: Path) -> str:
    """Hash a coarse quantized frame so harmless render noise can still reuse inference."""

    try:
        from PIL import Image

        image = Image.open(path).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
        quantized = bytes(int(pixel) // 16 for pixel in image.tobytes())
        return hashlib.sha256(quantized).hexdigest()
    except Exception:  # noqa: BLE001 - exact bytes remain a safe cache fallback
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""


def _visual_report_from_payload(payload: dict) -> VisualVerifierReport:
    recommendations = payload.get("recommendations", ())
    if isinstance(recommendations, str):
        recommendations = (recommendations,)
    else:
        recommendations = tuple(str(item) for item in recommendations)
    evidence = payload.get("evidence", {})
    return VisualVerifierReport(
        status=str(payload.get("status", "ok")),
        frame_path=str(payload.get("frame_path", "")),
        model_name=str(payload.get("model_name", "")),
        summary=str(payload.get("summary", "")),
        failure_type=str(payload.get("failure_type", "none")),
        confidence=float(payload.get("confidence", 0.0)),
        evidence=evidence if isinstance(evidence, dict) else {},
        recommendations=recommendations,
    )


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _require_route_discovery_global_camera(runtime_state: dict) -> None:
    """Fail before model inference when route discovery lacks a full-scene view."""

    camera = runtime_state.get("global_observation", {})
    if not isinstance(camera, dict):
        camera = {}
    available = bool(camera.get("available", False))
    source = str(camera.get("source", "")).strip().lower()
    locked = bool(camera.get("locked_to_camera", False))
    independent = bool(camera.get("independent_of_interactive_viewer", False))
    try:
        elevation = float(camera.get("elevation_deg", 0.0))
        distance = float(camera.get("distance_m", 0.0))
        age_s = float(camera.get("age_s", float("inf")))
    except (TypeError, ValueError):
        elevation = 0.0
        distance = 0.0
        age_s = float("inf")
    if (
        available
        and source == "mujoco_offscreen_fixed_camera"
        and locked
        and independent
        and elevation <= -89.0
        and distance >= 12.0
        and age_s <= 2.0
    ):
        return
    raise RuntimeError(
        "route-discovery execution requires Term1 --viewer-camera-preset global with "
        "a fresh fixed offscreen full-scene capture independent of the interactive viewer; "
        f"observed available={available} source={source or 'missing'} locked={locked} "
        f"independent={independent} elevation={elevation:.1f}deg "
        f"distance={distance:.1f}m age_s={age_s:.2f}"
    )


def _runtime_has_verified_payload_support(
    state: dict,
    *,
    safety_profile: str = "task",
) -> bool:
    """Recognize the persistent, physically verified initialization state."""

    safety = get_carry_safety_profile(safety_profile)
    if not state or str(state.get("runtime_status", "")) != "running":
        return False
    try:
        if time.time() - float(state.get("time_unix", 0.0)) > 2.0:
            return False
    except (TypeError, ValueError):
        return False
    control = state.get("control_health", {})
    if not isinstance(control, dict) or not bool(control.get("healthy", False)):
        return False
    if not bool(state.get("elastic_band_dropped", False)):
        return False
    payload = state.get("payload", {})
    if not isinstance(payload, dict) or not bool(payload.get("active", False)):
        return False
    contacts = payload.get("contacts", {})
    if not isinstance(contacts, dict):
        return False
    required_support_geometries = {
        "front_left_carry_support_plate",
        "front_right_carry_support_plate",
        "back_left_carry_support_plate",
        "back_right_carry_support_plate",
    }
    support_forces = contacts.get("support_forces", {})
    force_balanced = True
    if isinstance(support_forces, dict) and bool(support_forces.get("available", False)):
        force_by_robot = support_forces.get("normal_force_n_by_robot", {})
        force_by_robot = force_by_robot if isinstance(force_by_robot, dict) else {}
        force_balanced = (
            min(float(force_by_robot.get("front", 0.0)), float(force_by_robot.get("back", 0.0)))
            >= safety.min_support_force_per_robot_n
            and float(support_forces.get("balance_ratio", 0.0))
            >= safety.min_support_force_balance_ratio
        )
    support_target = payload.get("live_support_target", {})
    support_center_safe = safety.name != "strict"
    if safety.name == "strict" and isinstance(support_target, dict) and bool(
        support_target.get("tracking_available", support_target.get("ready", False))
    ):
        try:
            support_center_safe = math.hypot(
                float(payload.get("x", 0.0)) - float(support_target.get("x", 0.0)),
                float(payload.get("y", 0.0)) - float(support_target.get("y", 0.0)),
            ) <= safety.max_support_center_error_m
        except (TypeError, ValueError):
            support_center_safe = False
    robots = state.get("robots", {})
    pelvis_heights = [
        float(robot.get("pelvis_height_m", 0.0))
        for robot in robots.values()
        if isinstance(robot, dict)
    ] if isinstance(robots, dict) else []
    robot_pair_xy_separation_m = 0.0
    if isinstance(robots, dict):
        front = robots.get("front")
        back = robots.get("back")
        if isinstance(front, dict) and isinstance(back, dict):
            try:
                robot_pair_xy_separation_m = math.hypot(
                    float(front["x"]) - float(back["x"]),
                    float(front["y"]) - float(back["y"]),
                )
            except (KeyError, TypeError, ValueError):
                robot_pair_xy_separation_m = 0.0
    return (
        {"front", "back"}.issubset(set(contacts.get("support_robots", ())))
        and (
            not safety.require_all_support_pads_when_settled
            or required_support_geometries.issubset(
                set(contacts.get("support_geometries", ()))
            )
        )
        and not bool(contacts.get("floor", False))
        and (not safety.wall_contact_is_hard_failure or not bool(contacts.get("walls", ())))
        and float(payload.get("z", 0.0)) >= safety.min_payload_z_m
        and (
            safety.name != "strict"
            or float(payload.get("tilt_deg", 180.0)) <= safety.settled_max_tilt_deg
        )
        and bool(pelvis_heights)
        and min(pelvis_heights) >= safety.min_robot_pelvis_height_m
        and robot_pair_xy_separation_m >= safety.min_robot_xy_separation_m
        and force_balanced
        and support_center_safe
    )


def _checkpoint_requires_dual_support(state: dict) -> bool:
    """Return whether an accepted checkpoint represents an on-arms carry state."""

    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    if not isinstance(payload, dict) or not bool(payload.get("active", False)):
        return False
    contacts = payload.get("contacts", {})
    return isinstance(contacts, dict) and {"front", "back"}.issubset(
        set(contacts.get("support_robots", ()))
    )


def _require_restored_checkpoint_viable(
    live_state: dict,
    reference_state: dict,
    *,
    safety_profile: str,
) -> None:
    """Fail closed when restore did not recover a dispatchable trusted parent."""

    live = live_state if isinstance(live_state, dict) else {}
    reference = reference_state if isinstance(reference_state, dict) else {}
    if str(live.get("runtime_status", "")).strip() not in {"", "running"}:
        raise CheckpointBackendFatalError(
            "restored Term1 runtime is not running: "
            f"{live.get('runtime_status', 'unknown')}"
        )
    fault = live.get("runtime_fault", {})
    fault = fault if isinstance(fault, dict) else {}
    if str(fault.get("type", "")).strip() == "numerical_instability":
        raise CheckpointBackendFatalError(
            "restored checkpoint is numerically unstable: "
            f"reasons={fault.get('reasons', ())}"
        )
    nonfinite = _restored_state_nonfinite_fields(live)
    if nonfinite:
        raise CheckpointBackendFatalError(
            "restored checkpoint contains non-finite physical state: "
            + ", ".join(nonfinite)
        )
    if _checkpoint_requires_dual_support(reference) and not _runtime_has_verified_payload_support(
        live,
        safety_profile=safety_profile,
    ):
        raise CheckpointBackendFatalError(
            "restored checkpoint lost the trusted bilateral payload support"
        )


def _restored_state_nonfinite_fields(state: dict) -> list[str]:
    """Return pose fields that are not finite after checkpoint restore."""

    reasons: list[str] = []
    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    payload = payload if isinstance(payload, dict) else {}
    for key in ("x", "y", "z", "tilt_deg", "yaw_deg", "linear_speed_mps"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            reasons.append(f"payload.{key}")
    robots = state.get("robots", {}) if isinstance(state, dict) else {}
    robots = robots if isinstance(robots, dict) else {}
    for robot_id, robot in robots.items():
        robot = robot if isinstance(robot, dict) else {}
        pose = robot.get("root_pose", {})
        pose = pose if isinstance(pose, dict) else {}
        for key in ("x", "y", "z", "yaw_deg", "yaw_rad"):
            value = pose.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                reasons.append(f"robots.{robot_id}.root_pose.{key}")
        height = robot.get("pelvis_height_m")
        if isinstance(height, (int, float)) and not math.isfinite(float(height)):
            reasons.append(f"robots.{robot_id}.pelvis_height_m")
    return reasons


def _checkpoint_payload_is_target_supported(
    state: dict,
    *,
    allow_robot_contact: bool = False,
) -> bool:
    """Return whether a checkpoint is in the post-release target-support phase."""

    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    contacts = payload.get("contacts", {}) if isinstance(payload, dict) else {}
    return bool(
        isinstance(payload, dict)
        and bool(payload.get("active", False))
        and bool(payload.get("released", False))
        and isinstance(contacts, dict)
        and bool(contacts.get("target", False))
        and (
            allow_robot_contact
            or (
                not bool(contacts.get("robots", ()))
                and not bool(contacts.get("support_robots", ()))
            )
        )
        and not bool(contacts.get("floor", False))
        and not bool(contacts.get("walls", ()))
    )


def _observe_checkpoint_roundtrip_stability(
    *,
    runtime_state_file: Path,
    policy_action: str,
    stable_s: float,
    safety_profile: str,
) -> dict:
    """Verify a restored checkpoint with the active payload lifecycle contract."""

    stable_s = max(0.0, float(stable_s))
    if policy_action not in POST_RELEASE_CHECKPOINT_ACTIONS:
        return observe_payload_carry_stability(
            runtime_state_file,
            timeout_s=stable_s + 4.0,
            stable_s=stable_s,
            safety_profile=safety_profile,
        )

    started = time.monotonic()
    target_state = wait_for_payload_on_target(
        runtime_state_file,
        timeout_s=stable_s + 4.0,
        stable_s=stable_s,
        max_age_s=2.0,
        require_released=True,
        # The atomic release transaction may retain harmless residual arm
        # contact. Retreat/recovery checkpoints normally have none, and the
        # final task evaluator still requires the configured clearance.
        allow_robot_contact=policy_action == "release_payload",
    )
    status = read_payload_runtime_status(runtime_state_file, max_age_s=2.0)
    return {
        "success": True,
        "failure_type": "none",
        "reason": (
            "released payload remained physically supported by the target "
            "through the checkpoint round-trip stability window"
        ),
        "elapsed_s": time.monotonic() - started,
        "stable_duration_s": stable_s,
        "lifecycle": "post_release_target_support",
        "target_contact": bool(
            target_state.get("payload", {}).get("contacts", {}).get("target", False)
        ),
        "final_status": {
            "description": status.describe(),
            **_payload_status_payload(status),
        },
    }


def _wait_for_initial_physical_state(
    *,
    runtime_state_file: Path,
    timeout_s: float,
    stable_s: float,
    label: str,
    safety_profile: str = "task",
):
    """Accept a stable deferred placement gate or an already-supported payload."""

    safety = get_carry_safety_profile(safety_profile)
    current_state = read_json(runtime_state_file)
    if _checkpoint_payload_is_target_supported(
        current_state,
        allow_robot_contact=True,
    ):
        print(
            f"[cap-box-carry] waiting for {label} physical gate: "
            "fresh LowCmd and released payload stably supported by the target",
            flush=True,
        )
        wait_for_payload_on_target(
            runtime_state_file,
            timeout_s=max(0.0, float(timeout_s)),
            stable_s=max(0.0, float(stable_s)),
            max_age_s=2.0,
            require_released=True,
            allow_robot_contact=True,
        )
        status = read_payload_runtime_status(runtime_state_file, max_age_s=2.0)
        print(
            f"[cap-box-carry] {label} post-release target gate passed: "
            f"{status.describe()}",
            flush=True,
        )
        return status
    print(
        f"[cap-box-carry] waiting for {label} physical gate: "
        "fresh LowCmd, both robots upright, carry arms ready (payload may still be deferred)",
        flush=True,
    )
    status = wait_for_payload_initialization_ready(
        runtime_state_file,
        timeout_s=max(0.0, float(timeout_s)),
        stable_s=max(0.0, float(stable_s)),
        max_age_s=2.0,
        min_z_m=safety.min_payload_z_m,
        max_support_center_xy_error_m=safety.max_support_center_error_m,
        require_all_support_pads=safety.require_all_support_pads_when_settled,
        allow_non_support_robot_contacts=safety.name == "task",
        safety_profile=safety,
    )
    print(f"[cap-box-carry] {label} physical gate passed: {status.describe()}", flush=True)
    return status


def _payload_status_payload(status) -> dict:
    payload = asdict(status)
    payload["supporting_robots"] = sorted(status.supporting_robots)
    payload["supporting_geometries"] = sorted(status.supporting_geometries)
    return payload


def _turn_episode_preflight_report(
    error: str,
    *,
    policy_action: str,
    turn_episode: dict[str, object],
    visual_report: VisualVerifierReport | None = None,
    preflight_evidence: dict[str, object] | None = None,
) -> EvaluationReport:
    candidate_evidence = (
        dict(preflight_evidence)
        if isinstance(preflight_evidence, dict) and preflight_evidence
        else {}
    )
    evidence: dict[str, object] = {
        "turn_episode_preflight": {
            "physically_dispatched": False,
            "error": error,
            "active_episode": turn_episode,
            **(
                {"candidate_readiness": candidate_evidence}
                if candidate_evidence
                else {}
            ),
        }
    }
    if visual_report is not None:
        evidence["visual_verifier"] = asdict(visual_report)
    return annotate_evaluation_report(
        EvaluationReport(
            success=False,
            failure_type=(
                "turn_not_ready" if candidate_evidence else "alignment_regression"
            ),
            failed_subtask=policy_action,
            evidence=evidence,
            recommendations=(
                (
                    "replan from the exact candidate clearance evidence; the Host "
                    "does not select the replacement primitive, direction, roles, or parameters"
                    if candidate_evidence
                    else "regenerate the policy inside the active signed-turn continuity contract"
                ),
            ),
            visual_summary=(visual_report.summary if visual_report is not None else ""),
        )
    )


def _post_dispatch_model_terminal_report(
    preliminary_report: EvaluationReport,
    *,
    error: BaseException,
    failed_subtask: str,
    rollback_verification: dict[str, object],
) -> EvaluationReport:
    """Seal one dispatched transaction whose model-side evaluation could not finish."""

    terminal_reason = _terminal_reason_for_exception(error)
    evidence = {
        **preliminary_report.evidence,
        "post_execution_model_terminal": {
            "terminal_reason": terminal_reason,
            "error_type": type(error).__name__,
            "error": str(error),
            "physical_transaction_counted_once": True,
            "memory_promotion": False,
            "checkpoint_committed": False,
            "rollback_verification": rollback_verification,
        },
    }
    if not preliminary_report.success:
        return annotate_evaluation_report(
            replace(preliminary_report, evidence=evidence)
        )
    metrics = dict(preliminary_report.metrics)
    metrics["task_complete"] = 0.0
    return annotate_evaluation_report(
        replace(
            preliminary_report,
            success=False,
            failure_type="manual_review_required",
            failed_subtask=failed_subtask,
            metrics=metrics,
            evidence=evidence,
            recommendations=(
                "retry the invalid formal run from a fresh scene after the model "
                "provider or structured evaluator recovers",
            ),
            failure_level="system",
            diagnosis=(
                "The physical endpoint could not be promoted because post-execution "
                f"model evaluation terminated with {terminal_reason}."
            ),
            allowed_repairs=(),
            forbidden_repairs=("promote_memory", "commit_checkpoint"),
        )
    )


def _attach_simulation_observation(
    report: EvaluationReport,
    gate: dict[str, object],
    *,
    evidence_key: str,
    failed_subtask: str,
) -> EvaluationReport:
    """Attach a physics gate and prevent an unhealthy runtime from committing."""

    if not gate:
        return report
    evidence = {**report.evidence, evidence_key: gate}
    quality_warning = gate.get("quality_warning")
    if isinstance(quality_warning, dict) and quality_warning:
        warnings = evidence.get("physical_quality_warnings", [])
        warnings = list(warnings) if isinstance(warnings, list) else []
        warning = {**quality_warning, "source": evidence_key}
        if warning not in warnings:
            warnings.append(warning)
        evidence["physical_quality_warnings"] = warnings
    if report.success and not bool(gate.get("success", False)):
        failure_type = str(gate.get("failure_type", "")).strip()
        if not failure_type or failure_type == "none":
            failure_type = "runtime_state_missing"
        reason = str(gate.get("reason", "")).strip()
        recommendations = tuple(report.recommendations)
        if reason:
            recommendations = (
                f"restore runtime health before another physical rollout: {reason}",
                *recommendations,
            )
        return annotate_evaluation_report(
            replace(
                report,
                success=False,
                failure_type=failure_type,
                failed_subtask=failed_subtask,
                evidence=evidence,
                recommendations=recommendations,
            )
        )
    return annotate_evaluation_report(replace(report, evidence=evidence))


def _attach_checkpoint_roundtrip_semantics(
    report: EvaluationReport,
    roundtrip_report: EvaluationReport,
) -> EvaluationReport:
    """Audit restored semantics without running a second route acceptance gate.

    The fresh post-action observation has already been classified by
    ``transition_verifier``. Round-trip authority is limited to checkpoint and
    runtime integrity plus physical support; legacy route diagnostics retained
    here cannot overturn ACCEPT_SETUP or ACCEPT_PROGRESS.
    """

    roundtrip_payload = asdict(roundtrip_report)
    evidence = {
        **report.evidence,
        "checkpoint_roundtrip_semantic_evaluation": roundtrip_payload,
    }
    if report.success and not roundtrip_report.success:
        evidence["checkpoint_roundtrip_semantic_evaluation"] = {
            **roundtrip_payload,
            "authority": "diagnostic_only",
            "transition_outcome_preserved": transition_outcome_from_report(
                report
            ).value,
        }
    return annotate_evaluation_report(replace(report, evidence=evidence))


# Compatibility import for older offline tests and result-replay utilities.
_apply_stability_gate = _attach_simulation_observation


def _with_checkpoint_outcome(
    report: EvaluationReport,
    *,
    route_discovery: bool,
    checkpoint_committed: bool,
    checkpoint_before: str,
    checkpoint_after: str,
    rollback_performed: bool,
    physically_dispatched: bool,
) -> EvaluationReport:
    """Separate restorable physics state from route and task progress."""

    warnings = report.evidence.get("physical_quality_warnings", [])
    warnings = warnings if isinstance(warnings, list) else []
    warning_types = [
        str(item.get("type", "")).strip()
        for item in warnings
        if isinstance(item, dict) and str(item.get("type", "")).strip()
    ]
    transition_outcome = transition_outcome_from_report(report).value
    task_complete = bool(report.metrics.get("task_complete", 0.0))
    if not report.success:
        progress_status = "rejected"
        disposition = "restored_parent" if rollback_performed else "not_accepted"
    elif task_complete:
        progress_status = "task_complete"
        disposition = "committed_new" if checkpoint_committed else "retained_current"
    elif route_discovery:
        progress_status = (
            "route_progress"
            if transition_outcome == "ACCEPT_PROGRESS"
            else "safe_setup_only"
        )
        disposition = "committed_new" if checkpoint_committed else "retained_current"
    else:
        required = max(
            0.0,
            float(report.metrics.get("required_step_displacement_m", 0.0)),
        )
        target_progress = float(
            report.metrics.get("payload_target_progress_m", 0.0)
        )
        progress_status = (
            "goal_progress"
            if required > 0.0 and target_progress >= required
            else "safe_bounded_action_only"
        )
        disposition = "committed_new" if checkpoint_committed else "retained_current"
    outcome = {
        "physics_safe": bool(report.success),
        "checkpoint_disposition": disposition,
        "checkpoint_before": str(checkpoint_before),
        "checkpoint_after": str(checkpoint_after),
        "rollback_performed": bool(rollback_performed),
        "physically_dispatched": bool(physically_dispatched),
        "task_complete": task_complete,
        "transition_outcome": transition_outcome,
        "progress_status": progress_status,
        "phase_completion_proven": task_complete,
        "carry_quality": "degraded" if warning_types else "nominal",
        "quality_warning_types": warning_types,
        "interpretation": (
            "accepted means this physical state is restorable; it does not by "
            "itself prove route correctness or route-phase completion"
            if report.success and not task_complete
            else "the rollout was rejected and is not a trusted child state"
            if not report.success
            else "the final task completion condition is physically verified"
        ),
    }
    return replace(
        report,
        evidence={**report.evidence, "checkpoint_outcome": outcome},
    )


def _policy_action_method(
    policy: PolicyScript,
    *,
    validate_parameters: bool = True,
) -> str:
    """Return the one task-changing SDK call selected for this transaction."""

    if not policy.robot_scripts:
        return ""
    methods: list[str] = []
    for robot_script in policy.robot_scripts:
        method, _has_stop = validate_step_policy_code(
            robot_script.code,
            validate_parameters=validate_parameters,
        )
        methods.append(method)
    return next((method for method in methods if method != "hold_position"), methods[0])


def _policy_coordination_mode(policy: PolicyScript) -> str:
    modes = {
        str(script.coordination_mode).strip().lower() or "sync"
        for script in policy.robot_scripts
    }
    if not modes:
        return "sync"
    return next(iter(modes)) if len(modes) == 1 else "mixed"


def _policy_phase_id(policy: PolicyScript) -> str:
    phase_ids = {
        str(script.phase_id).strip()
        for script in policy.robot_scripts
        if str(script.phase_id).strip()
    }
    return next(iter(phase_ids)) if len(phase_ids) == 1 else ""


def _strategy_phase(
    task: BoxCarryTask,
    state: dict[str, object],
    *,
    accepted_steps: object = (),
) -> str:
    """Map trusted physics to a memory phase without selecting an action.

    In route-discovery runs the phase comes from the verifier-owned topology
    frontier. Model-authored ``task_plan`` labels and restorable setup
    checkpoints are deliberately excluded from this memory key.
    """

    payload = state.get("payload", {}) if isinstance(state, dict) else {}
    payload = payload if isinstance(payload, dict) else {}
    payload_x = _float_or_zero(payload.get("x"))
    payload_y = _float_or_zero(payload.get("y"))
    contacts = payload.get("contacts", {}) if isinstance(payload.get("contacts"), dict) else {}
    if task.route_discovery:
        if bool(contacts.get("target", False)):
            return (
                "target_release"
                if contacts.get("robots") or contacts.get("support_robots")
                else "robot_exit"
            )
        guard = trusted_route_phase_guard(
            task,
            state,
            accepted_steps=accepted_steps,
        )
        current_phase = str(guard.get("current_topological_phase", "")).strip()
        return current_phase or "route_discovery_transport"
    if task.task_stage == "stage1_straight":
        return "straight_transport"
    if task.task_stage in {"stage2_corridor", "stage3_right", "full_scene"}:
        if task.task_stage == "full_scene" and bool(contacts.get("target", False)):
            return (
                "target_release"
                if contacts.get("robots") or contacts.get("support_robots")
                else "robot_exit"
            )
        if payload_y >= task.stage2_corridor_goal_y_m - task.stage2_corridor_goal_tolerance_m:
            return "final_straight"
        if payload_x < task.stage1_goal_x_m - task.stage1_goal_tolerance_m:
            return "first_straight"
        return "long_corridor"
    target_distance = math.hypot(
        payload_x - task.target_center_x_m,
        payload_y - task.target_center_y_m,
    )
    return "target_alignment" if target_distance <= 1.0 else "straight_transport"


def _label_rollback_visual_timeline(
    failed_endpoint: VisualObservation,
    current_checkpoint: VisualObservation,
) -> tuple[VisualObservation, VisualObservation]:
    """Make failed and restored camera states impossible to confuse in model context."""

    failed = replace(
        failed_endpoint,
        source=f"failed_rollout_endpoint::{failed_endpoint.source}",
        summary=(
            "FAILED ROLLOUT ENDPOINT ONLY; this image is historical failure evidence and is "
            "not the state from which the next action executes. "
            f"{failed_endpoint.summary}"
        ).strip(),
    )
    current = replace(
        current_checkpoint,
        source=f"current_trusted_checkpoint::{current_checkpoint.source}",
        summary=(
            "CURRENT TRUSTED CHECKPOINT; the next action executes from this restored physical "
            f"state. {current_checkpoint.summary}"
        ).strip(),
    )
    return failed, current


def _restore_and_resynchronize(
    *,
    checkpoint_client: SimulatorCheckpointClient,
    backend: StaticSonicSkillBackend,
    checkpoint_id: str,
    checkpoint_reference_state: dict,
    attempt_dir: Path,
    heartbeat_file: Path,
    runtime_state_file: Path,
    idle_s: float,
    idle_keepalive: DualSonicIdleKeepalive | None = None,
    max_payload_xy_error_m: float,
    max_robot_xy_error_m: float,
    max_z_error_m: float,
    max_yaw_error_deg: float,
    safety_profile: str = "task",
    straight_heading_reference_rad: float | None = None,
    video_recorder: PhysicalActionVideoRecorder | None = None,
    video_attempt: int = 0,
    video_kind: str = "rollback",
    video_label: str = "",
    hold_planning: bool = False,
    candidate_checkpoint: bool = False,
) -> dict:
    """Restore physics, refresh SONIC's idle reference, then require stable support."""

    restore_settle_s = max(0.0, float(checkpoint_client.restore_settle_s))
    support_timeout_s = max(
        16.0,
        restore_settle_s + 8.0,
        float(idle_s) + 12.0,
    )

    if bool(getattr(backend, "execute", False)) and restore_settle_s < float(idle_s) + 3.0:
        raise RuntimeError(
            "checkpoint restore freeze is too short for the dual planner reset; require "
            "--checkpoint-resync-settle-s >= --checkpoint-resync-idle-s + 3"
        )

    # The restore event sets qpos/qvel and starts Term1's physics freeze in the
    # same simulation tick. No failed-state LowCmd can advance physics after this
    # acknowledgement.
    restore_video_segment = (
        video_recorder.start_segment(
            attempt=video_attempt,
            kind=video_kind,
            label=video_label or f"to_{checkpoint_id}",
            metadata={"checkpoint_id": checkpoint_id},
        )
        if video_recorder is not None
        else None
    )
    try:
        with _checkpoint_heartbeat_lease(idle_keepalive):
            restored_checkpoint = checkpoint_client.restore(
                checkpoint_id,
                hold_planning=hold_planning,
            )
    except BaseException as exc:
        if video_recorder is not None:
            video_recorder.stop_segment(
                restore_video_segment,
                status="restore_exception",
                tail_s=0.2,
                metadata={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    else:
        if video_recorder is not None:
            video_recorder.stop_segment(
                restore_video_segment,
                status="checkpoint_restored",
                tail_s=0.4,
            )
    # Both independent SONIC planners now observe the exact restored pose while
    # MuJoCo is frozen. One shared reset/IDLE barrier clears gait and heading
    # history without any visible rollback motion.
    with _paused_idle_keepalive(idle_keepalive):
        post_restore_resync = backend.resynchronize_idle(
            output_dir=attempt_dir / "checkpoint_post_restore_heading_resync",
            heartbeat_file=heartbeat_file,
            runtime_state_file=runtime_state_file,
            duration_s=max(0.5, float(idle_s)),
            stand_phase_id="checkpoint_post_restore_heading_resync",
        )
    write_json(
        attempt_dir / "checkpoint_post_restore_heading_resync.json",
        post_restore_resync,
    )
    with _checkpoint_heartbeat_lease(idle_keepalive):
        reasserted_checkpoint = checkpoint_client.restore(
            checkpoint_id,
            hold_planning=hold_planning,
        )
    _wait_for_checkpoint_resync_complete(
        runtime_state_file,
        checkpoint_id=checkpoint_id,
        timeout_s=support_timeout_s,
    )
    checkpoint_status = reasserted_checkpoint.runtime_state_after.get("checkpoint", {})
    persisted_reference = (
        checkpoint_status.get("reference_pose", {})
        if isinstance(checkpoint_status, dict)
        else {}
    )
    exact_reference_state = (
        persisted_reference
        if isinstance(persisted_reference, dict) and persisted_reference
        else reasserted_checkpoint.runtime_state_after or checkpoint_reference_state
    )
    reference_payload = exact_reference_state.get("payload", {})
    reference_payload_active = bool(
        reference_payload.get("active", False) if isinstance(reference_payload, dict) else False
    )
    try:
        if reference_payload_active and _checkpoint_payload_is_target_supported(
            exact_reference_state,
            allow_robot_contact=True,
        ):
            wait_for_payload_on_target(
                runtime_state_file,
                timeout_s=support_timeout_s,
                stable_s=1.0,
                max_age_s=2.0,
                require_released=True,
                allow_robot_contact=True,
            )
        elif reference_payload_active:
            wait_for_payload_supported(
                runtime_state_file,
                timeout_s=support_timeout_s,
                stable_s=1.0,
                max_age_s=2.0,
                allow_non_support_robot_contacts=safety_profile == "task",
                safety_profile=safety_profile,
            )
        else:
            wait_for_payload_initialization_ready(
                runtime_state_file,
                timeout_s=support_timeout_s,
                stable_s=1.0,
                max_age_s=2.0,
                safety_profile=safety_profile,
            )
    except TimeoutError as exc:
        if candidate_checkpoint:
            raise CandidateCheckpointRejectedError(
                "candidate checkpoint did not regain stable payload support"
            ) from exc
        raise
    restored_state = read_json(runtime_state_file)
    verification = _checkpoint_pose_verification(
        exact_reference_state,
        restored_state,
        max_payload_xy_error_m=max_payload_xy_error_m,
        max_robot_xy_error_m=max_robot_xy_error_m,
        max_z_error_m=max_z_error_m,
        max_yaw_error_deg=max_yaw_error_deg,
    )
    verification["restore_sequence"] = {
        "initial_restore_event_id": restored_checkpoint.event_id,
        "planner_resync_status": post_restore_resync.get("status", ""),
        "reassert_restore_event_id": reasserted_checkpoint.event_id,
        "order": "restore_planner_reset_reassert_release",
    }
    write_json(attempt_dir / "checkpoint_restore_verification.json", verification)
    print(
        "[cap-box-carry] rollback pose verification "
        f"payload_xy={verification['payload_xy_error_m']:.4f}m "
        f"max_robot_xy={verification['max_robot_xy_error_m']:.4f}m "
        f"max_yaw={verification['max_yaw_error_deg']:.2f}deg "
        f"within_tolerance={verification['within_tolerance']}",
        flush=True,
    )
    if not verification["within_tolerance"]:
        _raise_checkpoint_restore_drift(
            candidate_checkpoint=candidate_checkpoint,
            verification_path=(
                attempt_dir / "checkpoint_restore_verification.json"
            ),
        )
    try:
        _require_restored_checkpoint_viable(
            restored_state,
            exact_reference_state,
            safety_profile=safety_profile,
        )
    except CheckpointBackendFatalError as exc:
        if candidate_checkpoint:
            raise CandidateCheckpointRejectedError(str(exc)) from exc
        raise
    if straight_heading_reference_rad is not None and not hold_planning:
        verification["heading_normalization"] = (
            _normalize_straight_heading_after_restore(
                backend=backend,
                heartbeat_file=heartbeat_file,
                runtime_state_file=runtime_state_file,
                output_dir=attempt_dir / "checkpoint_heading_normalization",
                idle_keepalive=idle_keepalive,
                reference_yaw_rad=straight_heading_reference_rad,
                safety_profile=safety_profile,
            )
        )
        write_json(attempt_dir / "checkpoint_restore_verification.json", verification)
    return verification


def _normalize_straight_heading_after_restore(
    *,
    backend: StaticSonicSkillBackend,
    heartbeat_file: Path,
    runtime_state_file: Path,
    output_dir: Path,
    idle_keepalive: DualSonicIdleKeepalive | None,
    reference_yaw_rad: float,
    safety_profile: str,
) -> dict[str, object]:
    """Remove planner-state yaw drift before any restored carry state moves."""

    state = read_json(runtime_state_file)
    before = _straight_heading_status(state, reference_yaw_rad=reference_yaw_rad)
    result: dict[str, object] = {
        "required": not bool(before.get("within_tolerance", False)),
        "reference_heading_deg": math.degrees(reference_yaw_rad),
        "before": before,
        "attempts": [],
    }
    if not bool(before.get("available", False)):
        raise RuntimeError("restored state has no front/back yaw for heading normalization")
    if not _runtime_has_verified_payload_support(state, safety_profile=safety_profile):
        result.update(
            {
                "success": True,
                "skipped": True,
                "reason": "checkpoint does not contain an active dual-supported payload",
                "after": before,
            }
        )
        write_json(output_dir / "heading_normalization.json", result)
        return result
    if safety_profile != "strict" and not bool(before["within_tolerance"]):
        result.update(
            {
                "required": False,
                "success": True,
                "skipped": True,
                "diagnostic_only": True,
                "reason": (
                    "task-profile heading drift is verifier evidence for the Agent; "
                    "checkpoint save/restore must not inject a hidden corrective motion"
                ),
                "after": before,
            }
        )
        write_json(output_dir / "heading_normalization.json", result)
        return result
    if bool(before["within_tolerance"]):
        result.update({"success": True, "skipped": False, "after": before})
        write_json(output_dir / "heading_normalization.json", result)
        return result

    attempts: list[dict[str, object]] = []
    status = before
    for attempt_index in range(1, POST_RESTORE_HEADING_CORRECTION_LIMIT + 1):
        plan = _post_restore_heading_correction_plan(status)
        with _paused_idle_keepalive(idle_keepalive):
            record = backend.execute_team_actions(
                actions_by_robot=plan["actions"],
                phase_id=f"checkpoint_heading_normalization_{attempt_index:02d}",
                synchronization_group="checkpoint_heading_normalization",
                coordination_mode=str(plan["coordination_mode"]),
                output_dir=output_dir / f"attempt_{attempt_index:02d}",
                heartbeat_file=heartbeat_file,
                runtime_state_file=runtime_state_file,
                call_index=attempt_index,
            )
        stability = observe_payload_carry_stability(
            runtime_state_file,
            timeout_s=8.0,
            stable_s=2.0,
            safety_profile=safety_profile,
        )
        status = _straight_heading_status(
            read_json(runtime_state_file),
            reference_yaw_rad=reference_yaw_rad,
        )
        attempt_result = {
            "index": attempt_index,
            "method": plan["method"],
            "coordination_mode": plan["coordination_mode"],
            "actions": plan["actions"],
            "dispatch_status": record.status,
            "stability": stability,
            "heading_after": status,
        }
        attempts.append(attempt_result)
        if record.status != "success":
            raise RuntimeError(
                "post-restore heading correction dispatch failed: "
                f"{record.status}"
            )
        if not bool(stability.get("success", False)):
            raise RuntimeError(
                "post-restore heading correction lost stable payload support: "
                f"{stability.get('failure_type', 'unknown')}"
            )
        if bool(status.get("within_tolerance", False)):
            break

    result.update(
        {
            "success": bool(status.get("within_tolerance", False)),
            "skipped": False,
            "attempts": attempts,
            "after": status,
        }
    )
    write_json(output_dir / "heading_normalization.json", result)
    if not bool(result["success"]):
        raise RuntimeError(
            "restored carry heading did not converge before translation; "
            f"after={status}"
        )
    return result


def _preserve_resume_root_heading_report(
    state: dict,
    *,
    reference_yaw_rad: float,
) -> dict[str, object]:
    """Record a resumed episode root without changing its accepted physical pose."""

    status = _straight_heading_status(state, reference_yaw_rad=reference_yaw_rad)
    return {
        "required": False,
        "success": True,
        "skipped": True,
        "reason": (
            "explicit resumed checkpoint is the new episode root; preserve its accepted "
            "front/back pose and use its measured mean heading as the local reference"
        ),
        "reference_heading_deg": math.degrees(reference_yaw_rad),
        "before": status,
        "attempts": [],
        "after": status,
    }


def _straight_heading_status(
    state: dict,
    *,
    reference_yaw_rad: float,
) -> dict[str, object]:
    headings = _robot_headings_rad(state)
    if headings is None:
        return {"available": False, "within_tolerance": False}
    mean_heading = math.atan2(
        sum(math.sin(value) for value in headings.values()),
        sum(math.cos(value) for value in headings.values()),
    )
    errors_deg = {
        robot_id: math.degrees(_wrapped_angle(yaw - reference_yaw_rad))
        for robot_id, yaw in headings.items()
    }
    mean_error_deg = math.degrees(
        _wrapped_angle(mean_heading - reference_yaw_rad)
    )
    split_deg = abs(
        math.degrees(_wrapped_angle(headings["back"] - headings["front"]))
    )
    robots = state.get("robots", {}) if isinstance(state, dict) else {}
    lateral_positions: dict[str, float] = {}
    if isinstance(robots, dict):
        left_x = -math.sin(mean_heading)
        left_y = math.cos(mean_heading)
        for robot_id in ("front", "back"):
            robot = robots.get(robot_id, {})
            root_pose = robot.get("root_pose", {}) if isinstance(robot, dict) else {}
            pose = root_pose if isinstance(root_pose, dict) and root_pose else robot
            try:
                lateral_positions[robot_id] = (
                    float(pose["x"]) * left_x + float(pose["y"]) * left_y
                )
            except (KeyError, TypeError, ValueError):
                lateral_positions = {}
                break
    pivot_inner_robot_by_turn_direction = (
        {
            "counterclockwise": max(lateral_positions, key=lateral_positions.get),
            "clockwise": min(lateral_positions, key=lateral_positions.get),
        }
        if set(lateral_positions) == {"front", "back"}
        else {}
    )
    within_tolerance = bool(
        abs(mean_error_deg) <= MAX_STRAIGHT_HEADING_ERROR_DEG
        and split_deg <= MAX_STRAIGHT_HEADING_ERROR_DEG
    )
    return {
        "available": True,
        "within_tolerance": within_tolerance,
        "reference_heading_deg": math.degrees(reference_yaw_rad),
        "mean_heading_deg": math.degrees(mean_heading),
        "mean_heading_error_deg": mean_error_deg,
        "robot_heading_error_deg": errors_deg,
        "robot_heading_split_deg": split_deg,
        "robot_lateral_position_m": lateral_positions,
        "pivot_inner_robot_by_turn_direction": (
            pivot_inner_robot_by_turn_direction
        ),
        "max_heading_or_split_error_deg": MAX_STRAIGHT_HEADING_ERROR_DEG,
    }


def _post_restore_heading_correction_plan(
    status: dict[str, object],
) -> dict[str, object]:
    """Build one bounded correction from measured shared or relative yaw error."""

    if not bool(status.get("available", False)):
        raise RuntimeError("cannot plan heading correction without measured yaw")
    mean_error_deg = float(status.get("mean_heading_error_deg", 0.0))
    if abs(mean_error_deg) > MAX_STRAIGHT_HEADING_ERROR_DEG:
        angle_deg = min(15.0, max(5.0, abs(mean_error_deg)))
        direction = "counterclockwise" if mean_error_deg < 0.0 else "clockwise"
        inner_by_direction = status.get(
            "pivot_inner_robot_by_turn_direction",
            {},
        )
        if not isinstance(inner_by_direction, dict):
            inner_by_direction = {}
        inner_robot = str(inner_by_direction.get(direction, ""))
        if inner_robot not in {"front", "back"}:
            raise RuntimeError(
                "shared yaw correction requires measured front/back lateral geometry"
            )
        actions = {
            robot_id: {
                "method": "coordinated_pivot",
                "parameters": {
                    "turn_direction": direction,
                    "travel_direction": "forward",
                    "angle_deg": angle_deg,
                    "duration_s": 8.0,
                    "role": "inner" if robot_id == inner_robot else "outer",
                    "swing_radius_m": 0.8,
                },
            }
            for robot_id in ("front", "back")
        }
        return {
            "method": "coordinated_pivot",
            "coordination_mode": "async",
            "actions": actions,
        }

    errors = status.get("robot_heading_error_deg", {})
    if not isinstance(errors, dict) or not {"front", "back"}.issubset(errors):
        raise RuntimeError("relative yaw correction requires front/back heading errors")
    adjust_robot = max(errors, key=lambda robot_id: abs(float(errors[robot_id])))
    error_deg = float(errors[adjust_robot])
    delta_deg = min(15.0, max(5.0, abs(error_deg)))
    if error_deg > 0.0:
        delta_deg = -delta_deg
    actions = {
        robot_id: {
            "method": "coordinated_alignment",
            "parameters": {
                "axis": "yaw",
                "delta": delta_deg if robot_id == adjust_robot else 0.0,
                "duration_s": 6.0,
            },
        }
        for robot_id in ("front", "back")
    }
    return {
        "method": "coordinated_alignment",
        "coordination_mode": "async",
        "actions": actions,
    }


def _wait_for_checkpoint_resync_complete(
    runtime_state_file: Path,
    *,
    checkpoint_id: str,
    timeout_s: float,
    poll_s: float = 0.1,
) -> dict:
    """Wait until Term1 releases the exact-state checkpoint hold."""

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    last_state: dict = {}
    while True:
        last_state = read_json(runtime_state_file)
        checkpoint = last_state.get("checkpoint", {})
        if (
            isinstance(checkpoint, dict)
            and checkpoint.get("checkpoint_id") == checkpoint_id
            and not bool(checkpoint.get("resync_active", False))
        ):
            return last_state
        now = time.monotonic()
        if now >= deadline:
            observed_checkpoint_id = (
                checkpoint.get("checkpoint_id", "unknown")
                if isinstance(checkpoint, dict)
                else "unknown"
            )
            raise RuntimeError(
                "checkpoint rollback timed out before Term1 released exact-state "
                f"resynchronization for {checkpoint_id}; observed={observed_checkpoint_id}"
            )
        time.sleep(max(0.01, min(float(poll_s), deadline - now)))


def _build_branch_rewind_context(
    *,
    task: BoxCarryTask,
    checkpoint_id: str,
    lesson: str,
    runtime_state: dict,
    blocked_motion_objective: str = "",
    enforce_blocked_motion_objective: bool = False,
    blocked_primitive_families: tuple[str, ...] = (),
    retired_child_transaction_signature: str = "",
    retired_child_transaction: str = "",
    source: str = "verified_descendant_branch_dead_end",
) -> dict[str, object]:
    """Scope descendant failure evidence through one verified geometry repair."""

    payload = runtime_state.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    goal_x, goal_y = _active_goal_xy(task, payload)
    geometry = _body_frame_goal_alignment(
        runtime_state,
        {
            "active_stage_goal_x_m": goal_x,
            "active_stage_goal_y_m": goal_y,
        },
    )
    scores = geometry.get("candidate_goal_alignment", {})
    scores = scores if isinstance(scores, dict) else {}
    finite_scores = {
        str(direction): float(score)
        for direction, score in scores.items()
        if isinstance(score, (int, float)) and math.isfinite(float(score))
    }
    blocked_objective = str(blocked_motion_objective).strip()
    if not blocked_objective and finite_scores:
        best_direction, best_score = max(finite_scores.items(), key=lambda item: item[1])
        if best_score >= 0.35:
            blocked_objective = f"coordinated_linear_transport:{best_direction}"
    signed_turn_rewind = blocked_objective in {
        "coordinated_arc_transport:clockwise",
        "coordinated_arc_transport:counterclockwise",
        "coordinated_pivot:clockwise",
        "coordinated_pivot:counterclockwise",
    }
    blocked_families = sorted(
        {
            str(method).strip()
            for method in blocked_primitive_families
            if str(method).strip() in COORDINATION_PRIMITIVE_METHODS
        }
    )
    return {
        "active": True,
        "scope_checkpoint_id": checkpoint_id,
        "scope_active_goal_xy_m": [float(goal_x), float(goal_y)],
        "scope_team_heading_deg": team_heading_deg(runtime_state),
        "persistence_mode": (
            "until_route_turn_probe_success"
            if signed_turn_rewind
            else "until_route_objective_reoriented"
            if enforce_blocked_motion_objective
            else "until_next_accepted_checkpoint"
        ),
        # Route-regression evidence may explicitly exclude one exact semantic
        # objective at the restored checkpoint. Ordinary branch rewinds still
        # retire only the executable transaction.
        "blocked_motion_objective": (
            blocked_objective if enforce_blocked_motion_objective else ""
        ),
        "enforce_blocked_motion_objective": bool(
            enforce_blocked_motion_objective and blocked_objective
        ),
        "blocked_primitive_families": [],
        "retired_child_motion_objective": blocked_objective,
        "retired_child_primitive_families": blocked_families,
        "retired_child_transaction_signature": str(
            retired_child_transaction_signature
        ).strip(),
        "retired_child_transaction": str(retired_child_transaction).strip(),
        "source": str(source).strip() or "verified_descendant_branch_dead_end",
        "lesson": lesson,
        "selection_rule": (
            "Treat the retired child as a script-level counterexample. "
            + (
                f"The exact route-regressing objective {blocked_objective} is unavailable "
                "while it still points away from the recorded route frontier, but its "
                "primitive family remains available. "
                if enforce_blocked_motion_objective and blocked_objective
                else "The same primitive family or semantic direction remains selectable "
                "when current evidence supports it, but the exact verifier-rejected "
                "executable AST must not be replayed. "
            )
            + (
                "Keep the turn-rewrite context until a verified linear probe leaves the corner."
                if signed_turn_rewind
                else (
                    "The route exclusion expires when current body-frame geometry no "
                    "longer maps it away from the recorded route frontier."
                    if enforce_blocked_motion_objective
                    else "The rewrite context expires after the next accepted checkpoint."
                )
            )
        ),
    }


def _reconstruct_turn_episode_from_manifest(
    manifest: dict,
    *,
    checkpoint_id: str,
) -> dict:
    """Replay accepted turn evidence only up to the restored checkpoint."""

    checkpoint_id = str(checkpoint_id).strip()
    if not checkpoint_id or not isinstance(manifest, dict):
        return {}
    episode = empty_turn_episode()
    if checkpoint_id == str(manifest.get("initial_checkpoint", "")).strip():
        return episode
    for step in manifest.get("steps", ()):
        if not isinstance(step, dict):
            continue
        scripts_payload = step.get("robot_scripts", {})
        scripts_payload = (
            scripts_payload if isinstance(scripts_payload, dict) else {}
        )
        calls_payload = step.get("robot_calls", {})
        calls_payload = calls_payload if isinstance(calls_payload, dict) else {}
        robot_scripts: list[RobotPolicyScript] = []
        for robot_id in ("front", "back"):
            item = scripts_payload.get(robot_id, {})
            item = item if isinstance(item, dict) else {}
            call = calls_payload.get(robot_id, {})
            call = call if isinstance(call, dict) else {}
            code = str(item.get("code") or call.get("code") or "").strip()
            if not code:
                robot_scripts = []
                break
            robot_scripts.append(
                RobotPolicyScript(
                    robot_id=robot_id,
                    script_id=str(
                        item.get("script_id")
                        or f"{robot_id}_{step.get('script_id', 'accepted_step')}"
                    ),
                    code=code,
                    synchronization_group=str(
                        item.get("synchronization_group")
                        or f"replay_{step.get('index', 0)}"
                    ),
                    phase_id=str(item.get("phase_id", "")),
                    contact_intent=str(
                        item.get("contact_intent", "maintain_payload_support")
                    ),
                    coordination_mode=_persisted_coordination_mode(
                        item.get(
                            "coordination_mode",
                            step.get("coordination_mode", "sync"),
                        )
                    ),
                )
            )
        if len(robot_scripts) != 2:
            return {}
        policy = PolicyScript(
            script_id=str(step.get("script_id", "accepted_step_replay")),
            attempt=max(1, int(step.get("attempt", 1))),
            code=robot_scripts[0].code,
            robot_scripts=tuple(robot_scripts),
            verification_contract=dict(
                step.get("verification_contract", {})
                if isinstance(step.get("verification_contract"), dict)
                else {}
            ),
        )
        source_dir = Path(str(step.get("source_artifact_dir", "")))
        if source_dir and not source_dir.is_absolute():
            source_dir = REPO_ROOT / source_dir
        evaluation_payload = read_json(source_dir / "evaluation.json")
        metrics = evaluation_payload.get(
            "metrics",
            step.get("measured_outcome", {}),
        )
        report = EvaluationReport(
            success=True,
            failure_type="none",
            metrics=dict(metrics) if isinstance(metrics, dict) else {},
            evidence=dict(evaluation_payload.get("evidence", {}))
            if isinstance(evaluation_payload.get("evidence"), dict)
            else {},
        )
        visual_payload = read_json(source_dir / "visual_evaluation.json")
        visual_report = (
            _visual_report_from_payload(visual_payload)
            if visual_payload
            else None
        )
        step_checkpoint = str(step.get("checkpoint_after", "")).strip()
        episode = update_turn_episode(
            episode,
            policy=policy,
            report=report,
            visual_report=visual_report,
            checkpoint_id=step_checkpoint,
        )
        if step_checkpoint == checkpoint_id:
            episode["replayed_through_checkpoint_id"] = checkpoint_id
            return episode
    return {}


def _select_resume_turn_episode(
    *,
    reconstructed: dict[str, object],
    inherited: dict[str, object],
    checkpoint_id: str,
    resume_failed_attempt: bool,
) -> dict[str, object]:
    """Preserve same-checkpoint verifier rejection state on failed resumption."""

    if not resume_failed_attempt or not inherited:
        return reconstructed
    inherited_checkpoint = str(
        inherited.get("latest_checkpoint_id")
        or inherited.get("rejection_checkpoint_id")
        or ""
    ).strip()
    if inherited_checkpoint != str(checkpoint_id).strip():
        return reconstructed
    rejection_statuses = {
        "switch_allowed",
        "script_rewrite_required",
        "local_direction_replan_required",
        "local_execution_repair_required",
        "correction_required",
        "correction_retry_required",
    }
    has_explicit_rejection = bool(
        str(inherited.get("status", "")).strip() in rejection_statuses
        or any(
            inherited.get(key)
            for key in (
                "rejection_failure_type",
                "script_failure_type",
                "probe_retry_failure_type",
                "failed_executable",
                "rewrite_requirement",
            )
        )
    )
    if not has_explicit_rejection:
        return reconstructed
    return {
        **normalize_turn_episode(inherited),
        "state_source": "failed_attempt_parent_turn_episode",
    }


def _reopen_turn_episode_after_branch_rewind(
    episode: dict,
    *,
    checkpoint_id: str,
    reason: str,
    state_source: str = "accepted_episode_manifest_replay",
) -> dict:
    """Make restored turn state selectable after retiring a child branch."""

    restored = normalize_turn_episode(episode)
    restored.update(
        {
            "rewind_checkpoint_id": str(checkpoint_id),
            "rewind_reason": str(reason),
            "state_source": str(state_source),
        }
    )
    if (
        bool(restored.get("active", False))
        and str(restored.get("purpose", "")) == "route_turn"
    ):
        restored.update(
            {
                "status": "script_rewrite_required",
                "switch_allowed": True,
                "translation_pending": False,
                "latest_checkpoint_id": str(checkpoint_id),
                "rewrite_requirement": (
                    "select a materially different host-valid turn transaction "
                    "from the restored checkpoint"
                ),
            }
        )
    return restored


def _restore_checkpoint_and_wait_for_release(
    *,
    checkpoint_client: SimulatorCheckpointClient,
    checkpoint_id: str,
    runtime_state_file: Path,
    timeout_s: float,
):
    """Acknowledge one restore and finish its exact-state hold before continuing."""

    restored = checkpoint_client.restore(checkpoint_id)
    _wait_for_checkpoint_resync_complete(
        runtime_state_file,
        checkpoint_id=checkpoint_id,
        timeout_s=max(
            float(timeout_s),
            float(checkpoint_client.restore_settle_s) + 2.0,
        ),
    )
    return restored


def _checkpoint_pose_verification(
    reference_state: dict,
    restored_state: dict,
    *,
    max_payload_xy_error_m: float,
    max_robot_xy_error_m: float,
    max_z_error_m: float,
    max_yaw_error_deg: float,
) -> dict:
    reference_payload = reference_state.get("payload", {})
    restored_payload = restored_state.get("payload", {})
    if not isinstance(reference_payload, dict) or not isinstance(restored_payload, dict):
        raise RuntimeError("checkpoint pose verification requires payload state")

    payload_xy_error = math.hypot(
        float(restored_payload.get("x", 0.0)) - float(reference_payload.get("x", 0.0)),
        float(restored_payload.get("y", 0.0)) - float(reference_payload.get("y", 0.0)),
    )
    payload_z_error = abs(
        float(restored_payload.get("z", 0.0)) - float(reference_payload.get("z", 0.0))
    )
    payload_yaw_error_deg = abs(
        math.degrees(
            _wrapped_angle(
                float(restored_payload.get("yaw_rad", 0.0))
                - float(reference_payload.get("yaw_rad", 0.0))
            )
        )
    )

    reference_robots = reference_state.get("robots", {})
    restored_robots = restored_state.get("robots", {})
    robot_errors: dict[str, dict[str, float]] = {}
    for robot_id in ("front", "back"):
        reference = reference_robots.get(robot_id, {}) if isinstance(reference_robots, dict) else {}
        restored = restored_robots.get(robot_id, {}) if isinstance(restored_robots, dict) else {}
        if not isinstance(reference, dict) or not isinstance(restored, dict):
            raise RuntimeError(f"checkpoint pose verification requires {robot_id} robot state")
        robot_errors[robot_id] = {
            "xy_error_m": math.hypot(
                float(restored.get("x", 0.0)) - float(reference.get("x", 0.0)),
                float(restored.get("y", 0.0)) - float(reference.get("y", 0.0)),
            ),
            "z_error_m": abs(
                float(restored.get("z", 0.0)) - float(reference.get("z", 0.0))
            ),
            "yaw_error_deg": abs(
                math.degrees(
                    _wrapped_angle(
                        float(restored.get("yaw_rad", 0.0))
                        - float(reference.get("yaw_rad", 0.0))
                    )
                )
            ),
        }
    max_robot_xy_error = max(item["xy_error_m"] for item in robot_errors.values())
    max_robot_z_error = max(item["z_error_m"] for item in robot_errors.values())
    max_robot_yaw_error = max(item["yaw_error_deg"] for item in robot_errors.values())
    measured_max_z_error = max(payload_z_error, max_robot_z_error)
    measured_max_yaw_error = max(payload_yaw_error_deg, max_robot_yaw_error)
    tolerances = {
        "payload_xy_error_m": max(0.0, float(max_payload_xy_error_m)),
        "robot_xy_error_m": max(0.0, float(max_robot_xy_error_m)),
        "z_error_m": max(0.0, float(max_z_error_m)),
        "yaw_error_deg": max(0.0, float(max_yaw_error_deg)),
    }
    within_tolerance = (
        payload_xy_error <= tolerances["payload_xy_error_m"]
        and max_robot_xy_error <= tolerances["robot_xy_error_m"]
        and measured_max_z_error <= tolerances["z_error_m"]
        and measured_max_yaw_error <= tolerances["yaw_error_deg"]
    )
    return {
        "checkpoint_id": str(
            restored_state.get("checkpoint", {}).get("checkpoint_id", "")
            if isinstance(restored_state.get("checkpoint"), dict)
            else ""
        ),
        "within_tolerance": within_tolerance,
        "payload_xy_error_m": payload_xy_error,
        "payload_z_error_m": payload_z_error,
        "payload_yaw_error_deg": payload_yaw_error_deg,
        "max_robot_xy_error_m": max_robot_xy_error,
        "max_robot_z_error_m": max_robot_z_error,
        "max_robot_yaw_error_deg": max_robot_yaw_error,
        "max_z_error_m": measured_max_z_error,
        "max_yaw_error_deg": measured_max_yaw_error,
        "robots": robot_errors,
        "tolerances": tolerances,
    }


def _wrapped_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _load_straight_heading_references(
    path: Path,
    *,
    reset: bool,
) -> dict[str, float]:
    if reset:
        return {}
    payload = read_json(path)
    raw = payload.get("references_rad", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return {}
    references: dict[str, float] = {}
    for checkpoint_id, value in raw.items():
        try:
            references[str(checkpoint_id)] = _wrapped_angle(float(value))
        except (TypeError, ValueError):
            continue
    return references


def _write_straight_heading_references(
    path: Path,
    *,
    active_checkpoint_id: str,
    references: dict[str, float],
) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "active_checkpoint_id": str(active_checkpoint_id),
            "references_rad": {
                checkpoint_id: _wrapped_angle(value)
                for checkpoint_id, value in sorted(references.items())
            },
            "references_deg": {
                checkpoint_id: math.degrees(_wrapped_angle(value))
                for checkpoint_id, value in sorted(references.items())
            },
            "contract": (
                "translation checkpoints inherit their parent heading; only an accepted "
                "explicit turn may establish a new heading"
            ),
        },
    )


def _straight_heading_reference_for_checkpoint(
    checkpoint_id: str,
    *,
    references: dict[str, float],
    fallback_state: dict,
    runtime_state_file: Path,
    resume_as_episode_root: bool = False,
) -> float:
    if resume_as_episode_root:
        measured = _mean_robot_heading_rad(fallback_state)
        if measured is not None:
            return measured
    if checkpoint_id in references:
        return references[checkpoint_id]
    root_checkpoint = (
        Path(runtime_state_file).parent
        / "checkpoints"
        / "accepted_step_00.json"
    )
    root_payload = read_json(root_checkpoint)
    root_state = root_payload.get("physical_state", {})
    measured = _mean_robot_heading_rad(
        root_state if isinstance(root_state, dict) and root_state else fallback_state
    )
    return 0.0 if measured is None else measured


def _straight_heading_reference_after_accepted_action(
    *,
    current_reference_rad: float,
    trace: ExecutionTrace,
    runtime_state: dict,
) -> float:
    if not _trace_has_intentional_heading_change(trace):
        return current_reference_rad
    measured = _mean_robot_heading_rad(runtime_state)
    if measured is None:
        return current_reference_rad
    # A turn that returns the team inside the old straight envelope is a
    # correction, not a new route heading.
    if abs(math.degrees(_wrapped_angle(measured - current_reference_rad))) <= (
        MAX_STRAIGHT_HEADING_ERROR_DEG
    ):
        return current_reference_rad
    return measured


def _trace_has_intentional_heading_change(trace: ExecutionTrace) -> bool:
    for call in reversed(trace.skill_calls):
        if call.skill_name in {
            "carry_turn_in_place",
            "turn_clockwise",
            "turn_counterclockwise",
            "coordinated_arc_transport",
            "coordinated_pivot",
        }:
            return True
        if call.skill_name != "coordinated_alignment":
            continue
        if str(call.parameters.get("axis", "")).strip().lower() == "yaw":
            return True
        actions = call.parameters.get("actions", {})
        if isinstance(actions, dict) and any(
            isinstance(profile, dict)
            and isinstance(profile.get("parameters"), dict)
            and str(profile["parameters"].get("axis", "")).strip().lower()
            == "yaw"
            for profile in actions.values()
        ):
            return True
    return False


def _mean_robot_heading_rad(state: dict) -> float | None:
    headings = _robot_headings_rad(state)
    if headings is None:
        return None
    return math.atan2(
        sum(math.sin(value) for value in headings.values()),
        sum(math.cos(value) for value in headings.values()),
    )


def _robot_headings_rad(state: dict) -> dict[str, float] | None:
    robots = state.get("robots", {}) if isinstance(state, dict) else {}
    if not isinstance(robots, dict):
        return None
    headings: dict[str, float] = {}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id, {})
        if not isinstance(robot, dict):
            return None
        root_pose = robot.get("root_pose", {})
        pose = root_pose if isinstance(root_pose, dict) and root_pose else robot
        try:
            yaw = (
                float(pose["yaw_rad"])
                if "yaw_rad" in pose
                else math.radians(float(pose["yaw_deg"]))
            )
        except (KeyError, TypeError, ValueError):
            return None
        headings[robot_id] = yaw
    return headings


def _accepted_checkpoint_index(checkpoint_id: str) -> int:
    prefix = "accepted_step_"
    value = str(checkpoint_id).strip()
    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or not suffix.isdigit():
        raise ValueError(
            "--resume-checkpoint-id must use the accepted_step_N format, "
            f"got {checkpoint_id!r}"
        )
    return int(suffix)


def _checkpoint_parent_is_within_resume_root(
    *,
    parent_checkpoint_id: str,
    resume_checkpoint_id: str | None,
) -> bool:
    """Keep automatic branch search inside the explicitly resumed episode."""

    if resume_checkpoint_id is None:
        return True
    try:
        return _accepted_checkpoint_index(
            parent_checkpoint_id
        ) >= _accepted_checkpoint_index(resume_checkpoint_id)
    except ValueError:
        return False


def _require_visual_frame(observation, *, label: str) -> None:
    frame_path = Path(str(getattr(observation, "frame_path", "")))
    if frame_path.is_file():
        return
    source = str(getattr(observation, "source", "unknown"))
    summary = str(getattr(observation, "summary", "")).strip()
    raise RuntimeError(
        f"{label} requires a real external-camera frame; source={source!r} "
        f"frame_path={str(frame_path)!r} detail={summary or 'none'}. "
        "Formal model execution is stopped before dispatch instead of silently "
        "degrading to state-only planning."
    )


def _initial_accepted_step_counter(
    resume_checkpoint_id: str | None,
    *,
    next_accepted_step_index: int | None,
) -> int:
    if resume_checkpoint_id is None:
        if next_accepted_step_index is not None:
            raise ValueError("next accepted step requires a resumed checkpoint")
        return 0
    resumed_index = _accepted_checkpoint_index(resume_checkpoint_id)
    if next_accepted_step_index is None:
        return resumed_index
    if int(next_accepted_step_index) <= resumed_index:
        raise ValueError("next accepted step must be greater than resumed checkpoint")
    return int(next_accepted_step_index) - 1


def _manifest_source_artifacts_after_checkpoint(
    manifest: dict[str, object],
    checkpoint_id: str,
) -> tuple[str, ...]:
    """Return exact rollout sources retired when a manifest rewinds."""

    checkpoint_id = str(checkpoint_id).strip()
    initial_checkpoint = str(manifest.get("initial_checkpoint", "")).strip()
    steps = manifest.get("steps", ())
    steps = list(steps) if isinstance(steps, (list, tuple)) else []
    if checkpoint_id == initial_checkpoint:
        retired_steps = steps
    else:
        split_index: int | None = None
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            if str(step.get("checkpoint_after", "")).strip() == checkpoint_id:
                split_index = index + 1
                break
        if split_index is None:
            return ()
        retired_steps = steps[split_index:]
    return tuple(
        str(step.get("source_artifact_dir", "")).strip()
        for step in retired_steps
        if isinstance(step, dict) and str(step.get("source_artifact_dir", "")).strip()
    )


def _usage_summary(client, *, kinds: set[str] | None = None) -> dict:
    if client is not None and kinds is not None and hasattr(client, "call_records"):
        records = [
            item
            for item in getattr(client, "call_records", ())
            if str(item.get("kind", "")) in kinds
        ]
        completed_records = [
            item for item in records if str(item.get("request_id", "")).strip()
        ]
        return {
            "call_count": len(records),
            "input_tokens": sum(int(item.get("input_tokens", 0)) for item in records),
            "output_tokens": sum(int(item.get("output_tokens", 0)) for item in records),
            "cached_input_tokens": sum(
                int(item.get("cached_input_tokens", 0)) for item in records
            ),
            "image_input_tokens": sum(
                int(item.get("image_input_tokens", 0)) for item in records
            ),
            "reasoning_tokens": sum(
                int(item.get("reasoning_tokens", 0)) for item in records
            ),
            "latency_s": sum(float(item.get("latency_s", 0.0)) for item in records),
            "billed_call_count": sum(
                item.get("billing_amount_microusd") is not None
                for item in completed_records
            ),
            "unbilled_call_count": sum(
                item.get("billing_amount_microusd") is None
                for item in completed_records
            ),
            "failed_transport_round_count": len(records) - len(completed_records),
            "billing_amount_microusd": sum(
                int(item.get("billing_amount_microusd", 0) or 0)
                for item in records
            ),
            "billing_amount_usd": sum(
                int(item.get("billing_amount_microusd", 0) or 0)
                for item in records
            )
            / 1_000_000.0,
            "calls": records,
        }
    usage = getattr(client, "usage_summary", None)
    if callable(usage):
        return usage()
    return {"call_count": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "calls": []}


def _reconcile_billing_clients(*clients, total_timeout_s: float = 10.0) -> dict:
    """Run secret-safe deferred billing without delaying terminal sealing."""

    reports: list[dict] = []
    seen: set[int] = set()
    started = time.monotonic()
    deadline = started + max(0.1, float(total_timeout_s))
    for client in clients:
        if client is None or id(client) in seen:
            continue
        seen.add(id(client))
        reconcile = getattr(client, "reconcile_billing", None)
        if not callable(reconcile):
            reports.append(
                {
                    "model_name": str(getattr(client, "model_name", "")),
                    "status": "not_supported",
                    "initial_pending": 0,
                    "resolved": 0,
                    "remaining": 0,
                    "rounds": 0,
                }
            )
            continue
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            reports.append(
                {
                    "model_name": str(getattr(client, "model_name", "")),
                    "status": "incomplete",
                    "initial_pending": 0,
                    "resolved": 0,
                    "remaining": 1,
                    "rounds": 0,
                    "error_type": "BillingDeadlineExceeded",
                }
            )
            continue
        try:
            report = _call_billing_reconcile_bounded(reconcile, remaining_s)
        except Exception as exc:  # noqa: BLE001 - final artifacts must still be sealed
            report = {
                "status": "incomplete",
                "initial_pending": 0,
                "resolved": 0,
                "remaining": 1,
                "rounds": 0,
                "error_type": type(exc).__name__,
            }
        report = dict(report) if isinstance(report, dict) else {}
        reports.append(
            {
                "model_name": str(getattr(client, "model_name", "")),
                **report,
            }
        )
    remaining = sum(int(item.get("remaining", 0) or 0) for item in reports)
    statuses = {str(item.get("status", "")) for item in reports}
    return {
        "status": (
            "complete"
            if remaining == 0 and statuses <= {"complete", "not_required"}
            else "not_supported"
            if statuses == {"not_supported"}
            else "incomplete"
        ),
        "remaining": remaining,
        "total_timeout_s": float(total_timeout_s),
        "elapsed_s": time.monotonic() - started,
        "clients": reports,
    }


def _call_billing_reconcile_bounded(
    reconcile: Callable[..., object],
    timeout_s: float,
) -> object:
    """Call old or new billing clients without letting either block sealing."""

    result: dict[str, object] = {}

    def invoke() -> None:
        try:
            parameters = inspect.signature(reconcile).parameters
            if "total_timeout_s" in parameters:
                result["value"] = reconcile(
                    max_rounds=3,
                    initial_backoff_s=2.0,
                    per_request_delay_s=0.05,
                    total_timeout_s=timeout_s,
                    lookup_timeout_s=min(5.0, timeout_s),
                )
            else:
                result["value"] = reconcile()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            result["error"] = exc

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(max(0.1, float(timeout_s)))
    if thread.is_alive():
        raise TimeoutError("billing reconciliation exceeded terminal deadline")
    error = result.get("error")
    if isinstance(error, BaseException):
        raise error
    return result.get("value", {})


def _terminal_runtime_fault(state: object) -> dict[str, object]:
    """Return a machine-readable runtime fault present at decision termination."""

    if not isinstance(state, dict):
        return {
            "runtime_status": "missing",
            "control_healthy": False,
            "runtime_fault": {"type": "runtime_state_missing"},
        }
    runtime_status = str(state.get("runtime_status", "")).strip()
    control_health = state.get("control_health", {})
    control_health = control_health if isinstance(control_health, dict) else {}
    control_healthy = control_health.get("healthy")
    if runtime_status == "running" and control_healthy is True:
        return {}
    runtime_fault = state.get("runtime_fault", {})
    runtime_fault = runtime_fault if isinstance(runtime_fault, dict) else {}
    if not runtime_fault:
        runtime_fault = {
            "type": (
                "runtime_state_missing"
                if not runtime_status
                else "runtime_control_unhealthy"
            )
        }
    return {
        "runtime_status": runtime_status,
        "control_healthy": control_healthy,
        "runtime_fault": dict(runtime_fault),
    }


def _finalize_experiment(
    *,
    args,
    task,
    records,
    experiment_initial_state,
    experiment_started_at: float,
    codebase_guard,
    agent_client,
    visual_verifier,
    role_pipeline,
    evolution_controller,
    status: str = "",
    terminal_error: BaseException | None = None,
) -> dict:
    """Write one complete, auditable terminal artifact set exactly once."""

    # Capture the runtime at decision termination. Billing may legitimately take
    # several minutes and must not retroactively require idle backends to survive.
    terminal_runtime_state = read_json(args.runtime_state_file)
    terminal_runtime_fault = _terminal_runtime_fault(terminal_runtime_state)
    codebase_error: Exception | None = None
    try:
        codebase_guard.verify_unchanged(checkpoint="before_final_summary")
    except Exception as exc:
        codebase_error = exc

    write_json(
        args.output_dir / "final_summary.json",
        [asdict(record) for record in records],
    )
    vision_client = getattr(visual_verifier, "client", None)
    controller_metrics = _controller_metrics_with_dispatch_journals(
        evolution_controller.metrics(),
        args.output_dir,
    )
    task_succeeded = bool(controller_metrics.get("task_success", False))
    budget = controller_metrics.get("budget", {})
    budget = budget if isinstance(budget, dict) else {}
    terminal_status = str(status).strip() or (
        "SUCCESS"
        if task_succeeded
        else "PHYSICAL_BUDGET_EXHAUSTED"
        if bool(budget.get("exhausted", False))
        else "BACKEND_FATAL"
        if records and records[-1].evaluation.failure_type == "runtime_state_missing"
        else "SYSTEM_STALL"
    )
    operator_cancelled = terminal_status == "OPERATOR_CANCELLED"
    if codebase_error is not None and not operator_cancelled:
        terminal_status = "BACKEND_FATAL"
    if (
        not operator_cancelled
        and str(getattr(args, "eval_protocol", "")).strip()
        and terminal_runtime_fault
    ):
        terminal_status = "BACKEND_FATAL"
    runtime_fault_overrode_status = bool(
        terminal_status == "BACKEND_FATAL" and terminal_runtime_fault
    )

    # First seal a complete terminal decision with locally available usage.
    provisional_billing = {
        "status": "pending_reconciliation",
        "remaining": 0,
        "clients": [],
    }
    model_usage = _model_usage_snapshot(
        agent_client=agent_client,
        vision_client=vision_client,
        role_pipeline=role_pipeline,
        billing_reconciliation=provisional_billing,
    )
    metrics = _experiment_metrics(
        task=task,
        records=records,
        initial_state=experiment_initial_state,
        final_state=terminal_runtime_state,
        elapsed_s=time.time() - experiment_started_at,
        model_usage=model_usage,
        controller_metrics=controller_metrics,
    )
    metrics["terminal"] = _terminal_metrics_payload(
        terminal_status=terminal_status,
        terminal_error=terminal_error,
        runtime_fault_overrode_status=runtime_fault_overrode_status,
        terminal_runtime_fault=terminal_runtime_fault,
        codebase_error=codebase_error,
    )
    write_json(args.output_dir / "model_usage.json", model_usage)
    write_api_cost_ledger(
        args.output_dir,
        model_usage,
        experiment=_api_cost_experiment_metadata(args, terminal_status),
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_formal_run_manifest(
        args=args,
        repo_root=REPO_ROOT,
        codebase_snapshot=codebase_guard.snapshot,
        task=task,
        status=terminal_status,
        model_usage=model_usage,
    )

    # Billing is supplemental. Reconcile after the terminal artifacts exist,
    # then atomically refresh usage/metrics/manifest without changing outcome.
    billing_timeout_s = (
        FORMAL_BILLING_RECONCILIATION_TIMEOUT_S
        if str(getattr(args, "eval_protocol", "")).strip()
        else 10.0
    )
    try:
        billing_reconciliation = _reconcile_billing_clients(
            agent_client,
            vision_client,
            getattr(role_pipeline, "client", None),
            total_timeout_s=billing_timeout_s,
        )
    except BaseException as exc:  # noqa: BLE001 - billing cannot unseal a run
        billing_reconciliation = {
            "status": "incomplete",
            "remaining": 1,
            "total_timeout_s": float(billing_timeout_s),
            "elapsed_s": 0.0,
            "clients": [],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "terminal_outcome_preserved": True,
        }
    write_json(
        args.output_dir / "billing_reconciliation.json",
        billing_reconciliation,
    )
    model_usage = _model_usage_snapshot(
        agent_client=agent_client,
        vision_client=vision_client,
        role_pipeline=role_pipeline,
        billing_reconciliation=billing_reconciliation,
    )
    write_json(args.output_dir / "model_usage.json", model_usage)
    write_api_cost_ledger(
        args.output_dir,
        model_usage,
        experiment=_api_cost_experiment_metadata(args, terminal_status),
    )
    metrics = _experiment_metrics(
        task=task,
        records=records,
        initial_state=experiment_initial_state,
        final_state=terminal_runtime_state,
        elapsed_s=time.time() - experiment_started_at,
        model_usage=model_usage,
        controller_metrics=controller_metrics,
    )
    metrics["terminal"] = _terminal_metrics_payload(
        terminal_status=terminal_status,
        terminal_error=terminal_error,
        runtime_fault_overrode_status=runtime_fault_overrode_status,
        terminal_runtime_fault=terminal_runtime_fault,
        codebase_error=codebase_error,
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_formal_run_manifest(
        args=args,
        repo_root=REPO_ROOT,
        codebase_snapshot=codebase_guard.snapshot,
        task=task,
        status=terminal_status,
        model_usage=model_usage,
    )
    _mark_transaction_journals_terminal_metrics(
        args.output_dir,
        require_manifest=bool(str(getattr(args, "eval_protocol", "")).strip()),
    )
    if bool(getattr(args, "write_html", False)):
        _refresh_human_review(args.output_dir)
    return {
        "status": terminal_status,
        "metrics": metrics,
        "model_usage": model_usage,
    }


def _model_usage_snapshot(
    *, agent_client, vision_client, role_pipeline, billing_reconciliation: dict
) -> dict:
    return {
        "text": _usage_summary(
            agent_client,
            kinds={
                "text_generation",
                "vision_text_generation",
                "remote_text_generation",
            },
        ),
        "vision": _usage_summary(
            vision_client,
            kinds={"vision_generation", "remote_vision_generation"},
        ),
        "roles": (
            role_pipeline.usage_summary()
            if role_pipeline is not None
            else {
                "model_name": "",
                "shared_model_instance": False,
                "total_role_calls": 0,
                "roles": {},
            }
        ),
        "billing_reconciliation": dict(billing_reconciliation),
    }


def _api_cost_experiment_metadata(args, terminal_status: str) -> dict:
    return {
        "protocol_id": str(getattr(args, "eval_protocol", "")),
        "run_index": getattr(args, "run_index", None),
        "seed": getattr(args, "seed", None),
        "model": str(getattr(args, "planner_model", "")),
        "output_dir": str(getattr(args, "output_dir", "")),
        "terminal_status": str(terminal_status),
    }


def _terminal_metrics_payload(
    *,
    terminal_status: str,
    terminal_error: BaseException | None,
    runtime_fault_overrode_status: bool,
    terminal_runtime_fault: dict,
    codebase_error: Exception | None,
) -> dict:
    return {
        "status": terminal_status,
        "error_type": (
            "RuntimeControlLost"
            if runtime_fault_overrode_status
            else type(terminal_error).__name__ if terminal_error else ""
        ),
        "error": (
            "runtime was unhealthy at decision termination"
            if runtime_fault_overrode_status
            else str(terminal_error) if terminal_error else ""
        ),
        "decision_terminal_error_type": (
            type(terminal_error).__name__ if terminal_error else ""
        ),
        "decision_terminal_error": str(terminal_error) if terminal_error else "",
        "codebase_unchanged": codebase_error is None,
        "codebase_error": str(codebase_error) if codebase_error else "",
        "runtime_healthy_at_termination": not bool(terminal_runtime_fault),
        "runtime_fault_at_termination": terminal_runtime_fault,
    }


def _setup_recovery_metrics(physical_records) -> dict:
    """Audit setup-vs-progress efficiency without changing acceptance."""

    outcomes: list[str] = []
    setup_progress_m: list[float] = []
    for record in physical_records:
        if not bool(getattr(record.evaluation, "success", False)):
            continue
        outcome = transition_outcome_from_report(record.evaluation).value
        outcomes.append(outcome)
        if outcome == "ACCEPT_SETUP":
            try:
                setup_progress_m.append(
                    float(record.evaluation.metrics.get("payload_target_progress_m", 0.0))
                )
            except (TypeError, ValueError):
                setup_progress_m.append(0.0)
    setup_count = sum(1 for item in outcomes if item == "ACCEPT_SETUP")
    progress_count = sum(
        1 for item in outcomes if item in {"ACCEPT_PROGRESS", "TASK_COMPLETE"}
    )
    consecutive = 0
    max_consecutive = 0
    terminal_consecutive = 0
    for item in outcomes:
        if item == "ACCEPT_SETUP":
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
            terminal_consecutive = consecutive
        else:
            consecutive = 0
            terminal_consecutive = 0
    ratio = (
        float(setup_count) / float(progress_count)
        if progress_count
        else float(setup_count)
    )
    improvement = 0.0
    if len(setup_progress_m) >= 2:
        improvement = float(setup_progress_m[-1] - setup_progress_m[0])
    elif setup_progress_m:
        improvement = float(setup_progress_m[-1])
    return {
        "setup_count": setup_count,
        "progress_count": progress_count,
        "setup_to_progress_ratio": ratio,
        "consecutive_setup_count": terminal_consecutive,
        "max_consecutive_setup_count": max_consecutive,
        "recovery_objective_improvement": improvement,
    }


def _experiment_metrics(
    *,
    task,
    records,
    initial_state,
    final_state,
    elapsed_s: float,
    model_usage: dict,
    controller_metrics: dict | None = None,
) -> dict:
    controller_metrics = controller_metrics if isinstance(controller_metrics, dict) else {}
    task_complete = bool(
        (records and records[-1].evaluation.metrics.get("task_complete", 0.0))
        or controller_metrics.get("task_success", False)
    )
    physical_records = [
        record
        for record in records
        if trace_has_physical_dispatch(record.execution)
    ]
    accepted = sum(1 for record in physical_records if record.evaluation.success)
    rejected = len(physical_records) - accepted
    failures: dict[str, int] = {}
    for record in records:
        failure = record.evaluation.failure_type
        if failure != "none":
            failures[failure] = failures.get(failure, 0) + 1
    run_metrics = final_state.get("run_metrics", {}) if isinstance(final_state, dict) else {}
    path_length = float(run_metrics.get("payload_path_length_m", 0.0) or 0.0)
    initial_payload = initial_state.get("payload", {}) if isinstance(initial_state, dict) else {}
    start_x = float(initial_payload.get("x", task.target_center_x_m))
    start_y = float(initial_payload.get("y", task.target_center_y_m))
    straight_line = math.hypot(start_x - task.target_center_x_m, start_y - task.target_center_y_m)
    controller_physical_iterations = int(
        controller_metrics.get("physical_iterations", len(physical_records)) or 0
    )
    finalized_physical_iterations = len(physical_records)
    physical_iterations = max(
        finalized_physical_iterations,
        controller_physical_iterations,
    )
    pending_physical_iterations = max(
        0,
        physical_iterations - accepted - rejected,
    )
    accepted_route_reversal_count = 0
    for record in physical_records:
        quality = record.evaluation.evidence.get("accepted_route_quality", {})
        quality = quality if isinstance(quality, dict) else {}
        if (
            str(quality.get("classification", "")).strip()
            == "accepted_immediate_linear_reversal"
        ):
            accepted_route_reversal_count += 1
    setup_recovery = _setup_recovery_metrics(physical_records)
    iterations_to_success = controller_metrics.get(
        "iterations_to_success",
        physical_iterations if task_complete else None,
    )
    evolution_efficiency = float(
        controller_metrics.get(
            "evolution_efficiency",
            (1.0 / physical_iterations) if task_complete and physical_iterations else 0.0,
        )
        or 0.0
    )
    return {
        "task_success": task_complete,
        "task_success_rate": 1.0 if task_complete else 0.0,
        "elapsed_s": float(elapsed_s),
        "time_to_destination_s": float(elapsed_s) if task_complete else None,
        "executed_step_count": physical_iterations,
        "evolution_record_count": len(records),
        "pre_dispatch_record_count": len(records) - finalized_physical_iterations,
        "finalized_physical_record_count": finalized_physical_iterations,
        "pending_physical_evaluation_count": pending_physical_iterations,
        "physical_iterations": physical_iterations,
        "iterations_to_success": iterations_to_success,
        "evolution_efficiency": evolution_efficiency,
        "accepted_step_count": accepted,
        "rejected_step_count": rejected,
        "step_acceptance_rate": (
            accepted / physical_iterations if physical_iterations else 0.0
        ),
        "accepted_route_reversal_count": accepted_route_reversal_count,
        "setup_count": setup_recovery["setup_count"],
        "setup_to_progress_ratio": setup_recovery["setup_to_progress_ratio"],
        "consecutive_setup_count": setup_recovery["consecutive_setup_count"],
        "recovery_objective_improvement": setup_recovery[
            "recovery_objective_improvement"
        ],
        "setup_recovery": setup_recovery,
        "physical_accounting": {
            "accepted_physical": accepted,
            "rejected_physical": rejected,
            "accepted_plus_rejected": accepted + rejected,
            "pending_physical_evaluation": pending_physical_iterations,
            "controller_physical_iterations": controller_physical_iterations,
            "journal_confirmed_physical_iterations": int(
                controller_metrics.get(
                    "journal_confirmed_physical_iterations",
                    finalized_physical_iterations,
                )
                or 0
            ),
            "invariant_satisfied": bool(
                physical_iterations
                == accepted + rejected + pending_physical_iterations
                and physical_iterations == controller_physical_iterations
            ),
        },
        "rollback_count": sum(1 for record in records if record.rollback_performed),
        "attempts_to_task_success": iterations_to_success,
        "script_repair_count": rejected,
        "failure_counts": failures,
        "evolution_budget": controller_metrics.get("budget", {}),
        "strategy_improvement": controller_metrics.get("strategy_improvement", {}),
        "straight_line_payload_distance_m": straight_line,
        "payload_path_efficiency": straight_line / path_length if path_length > 1.0e-9 else 0.0,
        "physics": run_metrics,
        "phase_completion": _phase_completion_metrics(
            task,
            final_state,
            task_complete=task_complete,
            records=records,
        ),
        "model": {
            "llm_calls": int(model_usage["text"].get("call_count", 0)),
            "vlm_calls": int(model_usage["vision"].get("call_count", 0)),
            "input_tokens": int(model_usage["text"].get("input_tokens", 0))
            + int(model_usage["vision"].get("input_tokens", 0)),
            "output_tokens": int(model_usage["text"].get("output_tokens", 0))
            + int(model_usage["vision"].get("output_tokens", 0)),
            "cached_input_tokens": int(
                model_usage["text"].get("cached_input_tokens", 0)
            )
            + int(model_usage["vision"].get("cached_input_tokens", 0)),
            "image_input_tokens": int(
                model_usage["text"].get("image_input_tokens", 0)
            )
            + int(model_usage["vision"].get("image_input_tokens", 0)),
            "reasoning_tokens": int(
                model_usage["text"].get("reasoning_tokens", 0)
            )
            + int(model_usage["vision"].get("reasoning_tokens", 0)),
            "latency_s": float(model_usage["text"].get("latency_s", 0.0))
            + float(model_usage["vision"].get("latency_s", 0.0)),
            "billed_call_count": int(
                model_usage["text"].get("billed_call_count", 0)
            )
            + int(model_usage["vision"].get("billed_call_count", 0)),
            "unbilled_call_count": int(
                model_usage["text"].get("unbilled_call_count", 0)
            )
            + int(model_usage["vision"].get("unbilled_call_count", 0)),
            "billing_amount_microusd": int(
                model_usage["text"].get("billing_amount_microusd", 0)
            )
            + int(
                model_usage["vision"].get("billing_amount_microusd", 0)
            ),
            "billing_amount_usd": (
                int(model_usage["text"].get("billing_amount_microusd", 0))
                + int(
                    model_usage["vision"].get(
                        "billing_amount_microusd",
                        0,
                    )
                )
            )
            / 1_000_000.0,
            "billing_reconciliation_status": str(
                model_usage.get("billing_reconciliation", {}).get("status", "")
            ),
        },
    }


def _refresh_human_review(output_dir: Path) -> None:
    """Refresh derived review artifacts without affecting experiment validity."""

    try:
        build_run_review(output_dir)
    except Exception as exc:  # noqa: BLE001 - review output is never a physics gate
        print(
            "[cap-box-carry] warning: human review view could not be refreshed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def _phase_completion_metrics(
    task,
    final_state,
    *,
    task_complete: bool,
    records=(),
) -> dict:
    """Score the scene-configured topology from verifier-owned physics only."""

    total = len(task.route_progress_gates) + 2
    accepted_steps = [
        {
            "measured_outcome": dict(record.evaluation.metrics),
            "transition_outcome": transition_outcome_from_report(
                record.evaluation
            ).value,
            "positive_route_memory_eligible": transition_outcome_from_report(
                record.evaluation
            ).value
            in {"ACCEPT_PROGRESS", "TASK_COMPLETE"},
        }
        for record in records
        if record.evaluation.success
        and trace_has_physical_dispatch(record.execution)
    ]
    guard = trusted_route_phase_guard(
        task,
        final_state,
        accepted_steps=accepted_steps,
    )
    completed = int(guard.get("maximum_completed_prefix", 0) or 0)
    if task_complete:
        completed = total
    completed = min(total, max(0, completed))
    return {
        "completed": completed,
        "total": total,
        "fraction": completed / total,
        "current_topological_phase": str(
            guard.get("current_topological_phase", "")
        ),
        "source": "trusted_checkpoint_physics",
    }


if __name__ == "__main__":
    main()
