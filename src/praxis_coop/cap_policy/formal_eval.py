"""Immutable protocol and provenance helpers for formal CoEvolve evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


FORMAL_EVAL_PROTOCOL_ID = "coevolve-eval-v73-final"
FORMAL_EVAL_GIT_TAG = FORMAL_EVAL_PROTOCOL_ID
FORMAL_BENCHMARK_NAME = "backbone"
FORMAL_RUN_COUNT = 5
FORMAL_RUN_INDICES = tuple(range(FORMAL_RUN_COUNT))
FORMAL_SEEDS = tuple(range(FORMAL_RUN_COUNT))
FORMAL_PHYSICAL_BUDGET = 80
# Foundation-model calls are reported as cost/efficiency, not a task budget.
FORMAL_MODEL_ATTEMPT_BUDGET = 0
FORMAL_MAX_NEW_TOKENS = 8192
FORMAL_MAX_RESPONSES_PER_ROLE_INVOCATION = 5
FORMAL_MAX_PREDISPATCH_CYCLES_PER_CHECKPOINT = 3
FORMAL_TRANSACTION_JOURNAL_SCHEMA_VERSION = 2
FORMAL_TRANSACTION_JOURNAL_STAGES = (
    "dispatch_confirmed",
    "outcome",
    "controller_accounting",
    "checkpoint_lineage",
    "attempt_artifacts",
    "raw_rollout",
    "evolution_record",
    "memory_update",
    "terminal_metrics",
)
FORMAL_PROVIDER_TIMEOUT_S = 300.0
FORMAL_PROVIDER_NETWORK_RETRIES = 8
FORMAL_PROVIDER_KEEP_TERM4_ALIVE = True
FORMAL_PROVIDER_RECOVERY_BACKOFF_S = 60.0
FORMAL_PROVIDER_RECOVERY_WALL_TIME_CAP_S = 1800.0
# Terminal metrics/manifest are sealed before billing. Keep reconciliation
# bounded so unavailable provider accounting cannot hold a finished run open.
FORMAL_BILLING_RECONCILIATION_TIMEOUT_S = 10.0
FORMAL_TASK_ID = "two_g1_long_box_carry_manual_layout"
FORMAL_TASK_TARGET = (
    "Two G1 robots carry the long_box through the authored scene and "
    "stably place it on the red target cylinder."
)
FORMAL_ROUTE_SPEC_RELATIVE_PATH = Path(
    "configs/routes/two_g1_box_push_layout.json"
)
FORMAL_PAYLOAD_MASS_KG = 0.5
FORMAL_PAYLOAD_REARWARD_OFFSET_M = 0.10
FORMAL_UPPER_BODY_POSE = "elbow60_hand60_carry_ready"
FORMAL_UPPER_BODY_KP = 24.0
FORMAL_UPPER_BODY_KD = 1.5
FORMAL_UPPER_BODY_RAMP_S = 6.0
FORMAL_PALM_TILT_DEG = 60.0
FORMAL_MUJOCO_SOLVER = "newton"
FORMAL_TARGET_CENTER_X_M = -1.5
FORMAL_TARGET_CENTER_Y_M = 7.6
FORMAL_TARGET_CENTER_Z_M = 0.3
FORMAL_TARGET_RADIUS_M = 0.3
FORMAL_TARGET_HALF_HEIGHT_M = 0.3
FORMAL_TARGET_XY_TOLERANCE_M = 0.18
FORMAL_TARGET_Z_TOLERANCE_M = 0.08
FORMAL_MAX_TERMINAL_PAYLOAD_TILT_DEG = 20.0
FORMAL_ROBOT_EXIT_MIN_DISTANCE_M = 1.10
FORMAL_TERMINAL_STABLE_S = 1.0
FORMAL_RECORD_ACTION_VIDEOS = False
FORMAL_ACTION_VIDEO_FPS = 10.0
FORMAL_ARGUMENT_OVERRIDES: Mapping[str, Any] = {
    "planner_timeout_s": FORMAL_PROVIDER_TIMEOUT_S,
    "planner_network_retries": FORMAL_PROVIDER_NETWORK_RETRIES,
    "keep_term4_on_transient_provider_error": FORMAL_PROVIDER_KEEP_TERM4_ALIVE,
    "provider_recovery_backoff_s": FORMAL_PROVIDER_RECOVERY_BACKOFF_S,
    "provider_recovery_wall_time_cap_s": (
        FORMAL_PROVIDER_RECOVERY_WALL_TIME_CAP_S
    ),
    "task_id": FORMAL_TASK_ID,
    "target": FORMAL_TASK_TARGET,
    "task_stage": "full_scene",
    "route_discovery": True,
    "route_spec": FORMAL_ROUTE_SPEC_RELATIVE_PATH.as_posix(),
    "max_attempts": FORMAL_MODEL_ATTEMPT_BUDGET,
    "iteration_budget": FORMAL_PHYSICAL_BUDGET,
    "strategy_failure_budget": 0,
    "require_robots_exit": True,
    "physical_safety_profile": "task",
    "temperature": 0.2,
    "max_new_tokens": FORMAL_MAX_NEW_TOKENS,
    "vl_temperature": 0.0,
    "vl_max_new_tokens": 1536,
    "checkpoint_timeout_s": 10.0,
    "checkpoint_resync_settle_s": 4.0,
    "checkpoint_resync_idle_s": 1.0,
    "initial_support_timeout_s": 120.0,
    "initial_support_stable_s": 1.0,
    "start_pulse_seconds": 0.0,
    "stand_seconds": 0.5,
    "bootstrap_start_pulse_seconds": 2.0,
    "bootstrap_idle_seconds": 2.0,
    "transport_skill": "carry_forward",
    "transport_distance_m": 1.0,
    "transport_speed_mps": 0.50,
    "minimum_transport_window_s": 1.0,
    "minimum_lateral_transport_window_s": 0.4,
    "transport_duration_gain": 1.0,
    "transport_duration_baseline_m": 2.0,
    "minimum_transport_ramp_s": 1.5,
    "post_transport_idle_s": 1.0,
    "transport_ramp_s": 1.5,
    "post_action_stable_s": FORMAL_TERMINAL_STABLE_S,
    "post_action_stability_timeout_s": 5.0,
    "precommit_stable_s": FORMAL_TERMINAL_STABLE_S,
    "precommit_stability_timeout_s": 5.0,
    "no_team_sync": False,
    "team_sync_active_catchup": True,
    "team_sync_longitudinal_kp": 2.0,
    "team_sync_cross_track_kp": 0.6,
    "team_sync_payload_kp": 1.2,
    "team_sync_error_filter_alpha": 0.35,
    "team_sync_pause_leader_error_m": 0.008,
    "team_sync_resume_error_m": 0.003,
    "team_sync_lagging_boost_mps": 0.06,
    "team_sync_max_correction_mps": 0.06,
    "team_sync_min_active_speed_ratio": 0.0,
    "no_require_payload_on_target": False,
    "camera_device": None,
    "visual_frame": None,
    "record_action_videos": FORMAL_RECORD_ACTION_VIDEOS,
    "action_video_fps": FORMAL_ACTION_VIDEO_FPS,
    "visual_window_name": "MuJoCo : praxis_manual_layout_two_g1_43dof_scene",
}
SONIC_RUNTIME_ARTIFACTS = (
    "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/target/release/"
    "g1_deploy_onnx_ref",
    "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/"
    "model_encoder.onnx",
    "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/"
    "model_decoder.onnx",
    "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/"
    "observation_config.yaml",
    "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/"
    "planner_sonic.onnx",
    "third_party/GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/"
    "unitree_sdk2py_bridge.py",
)


class FormalEvalConfigurationError(RuntimeError):
    """Raised when a requested formal run violates the frozen protocol."""


@dataclass(frozen=True)
class FormalBackbone:
    key: str
    requested_model: str
    provider: str = "openai_compatible"
    planner_backend: str = "openai_compatible"


FORMAL_BACKBONES = (
    FormalBackbone(
        key="gpt-5.4-mini-2026-03-17",
        requested_model="gpt-5.4-mini-2026-03-17",
    ),
    FormalBackbone(
        key="gemini-3.5-flash",
        requested_model="gemini-3.5-flash",
    ),
    FormalBackbone(
        key="claude-sonnet-4-6",
        requested_model="claude-sonnet-4-6",
    ),
)
FORMAL_BACKBONES_BY_MODEL = {item.requested_model: item for item in FORMAL_BACKBONES}
FORMAL_BACKBONES_BY_KEY = {item.key: item for item in FORMAL_BACKBONES}


@dataclass(frozen=True)
class FormalEvalProtocol:
    protocol_id: str = FORMAL_EVAL_PROTOCOL_ID
    benchmark: str = FORMAL_BENCHMARK_NAME
    run_count: int = FORMAL_RUN_COUNT
    physical_budget: int = FORMAL_PHYSICAL_BUDGET
    model_attempt_budget: int = FORMAL_MODEL_ATTEMPT_BUDGET
    max_responses_per_role_invocation: int = (
        FORMAL_MAX_RESPONSES_PER_ROLE_INVOCATION
    )
    max_predispatch_cycles_per_checkpoint: int = (
        FORMAL_MAX_PREDISPATCH_CYCLES_PER_CHECKPOINT
    )
    provider_network_retries: int = FORMAL_PROVIDER_NETWORK_RETRIES
    keep_term4_on_transient_provider_error: bool = (
        FORMAL_PROVIDER_KEEP_TERM4_ALIVE
    )
    provider_recovery_backoff_s: float = FORMAL_PROVIDER_RECOVERY_BACKOFF_S
    provider_recovery_wall_time_cap_s: float = (
        FORMAL_PROVIDER_RECOVERY_WALL_TIME_CAP_S
    )
    billing_reconciliation_timeout_s: float = (
        FORMAL_BILLING_RECONCILIATION_TIMEOUT_S
    )
    system_stall_is_valid_failure: bool = True
    memory_condition: str = "empty_online_memory"
    payload_mass_kg: float = FORMAL_PAYLOAD_MASS_KG
    payload_rearward_offset_m: float = FORMAL_PAYLOAD_REARWARD_OFFSET_M
    upper_body_pose: str = FORMAL_UPPER_BODY_POSE
    upper_body_kp: float = FORMAL_UPPER_BODY_KP
    upper_body_kd: float = FORMAL_UPPER_BODY_KD
    upper_body_ramp_s: float = FORMAL_UPPER_BODY_RAMP_S
    palm_tilt_deg: float = FORMAL_PALM_TILT_DEG
    mujoco_solver: str = FORMAL_MUJOCO_SOLVER
    target_center_x_m: float = FORMAL_TARGET_CENTER_X_M
    target_center_y_m: float = FORMAL_TARGET_CENTER_Y_M
    target_center_z_m: float = FORMAL_TARGET_CENTER_Z_M
    target_radius_m: float = FORMAL_TARGET_RADIUS_M
    target_half_height_m: float = FORMAL_TARGET_HALF_HEIGHT_M
    target_xy_tolerance_m: float = FORMAL_TARGET_XY_TOLERANCE_M
    target_z_tolerance_m: float = FORMAL_TARGET_Z_TOLERANCE_M
    max_terminal_payload_tilt_deg: float = FORMAL_MAX_TERMINAL_PAYLOAD_TILT_DEG
    robot_exit_min_distance_m: float = FORMAL_ROBOT_EXIT_MIN_DISTANCE_M
    terminal_stable_s: float = FORMAL_TERMINAL_STABLE_S
    record_action_videos: bool = FORMAL_RECORD_ACTION_VIDEOS
    action_video_fps: float = FORMAL_ACTION_VIDEO_FPS


PROTOCOL = FormalEvalProtocol()


def formal_backbone(model_or_key: str) -> FormalBackbone:
    """Resolve one frozen backbone by its exact model id or directory key."""

    value = str(model_or_key).strip()
    model = FORMAL_BACKBONES_BY_MODEL.get(value) or FORMAL_BACKBONES_BY_KEY.get(value)
    if model is None:
        allowed = ", ".join(item.requested_model for item in FORMAL_BACKBONES)
        raise FormalEvalConfigurationError(
            f"model {value!r} is not in {FORMAL_EVAL_PROTOCOL_ID}; choose one of: "
            f"{allowed}"
        )
    return model


def formal_run_output_dir(
    repo_root: Path,
    *,
    model_or_key: str,
    run_index: int,
) -> Path:
    """Return the only valid result directory for one backbone run."""

    model = formal_backbone(model_or_key)
    index = int(run_index)
    if index not in FORMAL_RUN_INDICES:
        raise FormalEvalConfigurationError(
            f"formal run_index must be one of {FORMAL_RUN_INDICES}; got {index}"
        )
    return (
        Path(repo_root)
        / "results"
        / "deployment"
        / "formal"
        / FORMAL_EVAL_PROTOCOL_ID
        / FORMAL_BENCHMARK_NAME
        / model.key
        / f"seed_{FORMAL_SEEDS[index]:03d}"
    )


def formal_memory_paths(output_dir: Path) -> dict[str, Path]:
    """Return run-local online-memory paths used by every formal run."""

    memory_dir = Path(output_dir) / "memory"
    return {
        "raw_attempt_memory": memory_dir / "raw_rollouts.jsonl",
        "strategy_memory": memory_dir / "strategy_specs.json",
        "action_exemplar_memory": memory_dir / "action_exemplars.json",
        "episode_manifest": memory_dir / "accepted_episode_manifest.json",
    }


def configure_formal_args(
    args: Any,
    *,
    repo_root: Path,
    default_output_dir: Path,
) -> Any:
    """Apply run-local defaults and enforce the immutable formal protocol."""

    output_dir = Path(args.output_dir)
    if not str(getattr(args, "eval_protocol", "")).strip():
        _set_run_local_memory_paths(args, output_dir)
        return args
    if args.eval_protocol != FORMAL_EVAL_PROTOCOL_ID:
        raise FormalEvalConfigurationError(
            f"unknown evaluation protocol {args.eval_protocol!r}"
        )
    if args.run_index is None:
        raise FormalEvalConfigurationError("formal evaluation requires --run-index")
    model = formal_backbone(args.planner_model)
    expected_output = formal_run_output_dir(
        repo_root,
        model_or_key=model.key,
        run_index=args.run_index,
    )
    if output_dir.resolve(strict=False) == Path(default_output_dir).resolve(
        strict=False
    ):
        output_dir = expected_output
        args.output_dir = expected_output
    elif output_dir.resolve(strict=False) != expected_output.resolve(strict=False):
        raise FormalEvalConfigurationError(
            "formal output directory must be exactly "
            f"{expected_output}; got {output_dir}"
        )
    expected_seed = FORMAL_SEEDS[int(args.run_index)]
    if args.seed is None:
        args.seed = expected_seed
    if int(args.seed) != expected_seed:
        raise FormalEvalConfigurationError(
            f"run_index={args.run_index} requires seed={expected_seed}; got {args.seed}"
        )
    if not bool(args.execute):
        raise FormalEvalConfigurationError("formal evaluation requires --execute")
    if args.agent_mode != "model" or args.visual_evaluator != "model":
        raise FormalEvalConfigurationError(
            "formal evaluation requires model agent mode and model visual evaluation"
        )
    if args.planner_backend != model.planner_backend:
        raise FormalEvalConfigurationError(
            f"formal model {model.requested_model} requires "
            f"--planner-backend {model.planner_backend}"
        )
    if bool(args.allow_dirty_codebase):
        raise FormalEvalConfigurationError(
            "--allow-dirty-codebase is forbidden in formal evaluation"
        )
    forbidden_resume_values = {
        "resume_policy": args.resume_policy,
        "resume_checkpoint_id": args.resume_checkpoint_id,
        "resume_attempt_dir": args.resume_attempt_dir,
        "resume_failed_attempt_dir": args.resume_failed_attempt_dir,
        "next_accepted_step_index": args.next_accepted_step_index,
        "next_policy_attempt_index": args.next_policy_attempt_index,
        "branch_rewind_note": str(args.branch_rewind_note).strip(),
        "branch_rewind_route_exclusion": str(
            getattr(args, "branch_rewind_route_exclusion", "")
        ).strip(),
    }
    active_resume = sorted(
        key
        for key, value in forbidden_resume_values.items()
        if value is not None and value != ""
    )
    if active_resume:
        raise FormalEvalConfigurationError(
            "formal evaluation forbids resume, operator rewind, and manual branch "
            f"selection arguments: {active_resume}"
        )
    if bool(args.stop_before_physical_dispatch):
        raise FormalEvalConfigurationError(
            "formal evaluation cannot stop before physical dispatch"
        )
    if bool(args.no_require_payload_on_target) or bool(args.no_team_sync):
        raise FormalEvalConfigurationError(
            "formal evaluation requires target placement and the frozen team-sync layer"
        )

    runtime_default = Path(default_output_dir) / "runtime_state.json"
    heartbeat_default = Path(default_output_dir) / "planner_heartbeat.json"
    runtime_path = Path(args.runtime_state_file)
    heartbeat_path = Path(args.heartbeat_file)
    if runtime_path.resolve(strict=False) == runtime_default.resolve(strict=False):
        args.runtime_state_file = output_dir / "runtime_state.json"
    elif runtime_path.resolve(strict=False) != (
        output_dir / "runtime_state.json"
    ).resolve(strict=False):
        raise FormalEvalConfigurationError(
            "formal --runtime-state-file must be <output-dir>/runtime_state.json"
        )
    if heartbeat_path.resolve(strict=False) == heartbeat_default.resolve(strict=False):
        args.heartbeat_file = output_dir / "planner_heartbeat.json"
    elif heartbeat_path.resolve(strict=False) != (
        output_dir / "planner_heartbeat.json"
    ).resolve(strict=False):
        raise FormalEvalConfigurationError(
            "formal --heartbeat-file must be <output-dir>/planner_heartbeat.json"
        )

    for attribute, value in FORMAL_ARGUMENT_OVERRIDES.items():
        setattr(args, attribute, value)
    args.route_spec = Path(repo_root) / FORMAL_ROUTE_SPEC_RELATIVE_PATH
    _set_run_local_memory_paths(args, output_dir, require_exact=True)
    return args


def _set_run_local_memory_paths(
    args: Any,
    output_dir: Path,
    *,
    require_exact: bool = False,
) -> None:
    expected = formal_memory_paths(output_dir)
    for attribute, expected_path in expected.items():
        current = getattr(args, attribute, None)
        if current is None:
            setattr(args, attribute, expected_path)
            continue
        if require_exact and Path(current).resolve(
            strict=False
        ) != expected_path.resolve(strict=False):
            raise FormalEvalConfigurationError(
                f"formal --{attribute.replace('_', '-')} must be {expected_path}"
            )


def validate_fresh_formal_run(args: Any) -> None:
    """Reject memory reuse, prior attempts, and completed artifacts before startup."""

    if not str(getattr(args, "eval_protocol", "")).strip():
        return
    output_dir = Path(args.output_dir)
    memory_paths = formal_memory_paths(output_dir)
    stale_paths = [path for path in memory_paths.values() if path.exists()]
    stale_paths.extend(sorted(output_dir.glob("attempt_*")))
    stale_paths.extend(
        path
        for name in (
            "run_manifest.json",
            "final_summary.json",
            "metrics.json",
            "raw_rollout_evidence.jsonl",
        )
        if (path := output_dir / name).exists()
    )
    if stale_paths:
        rendered = ", ".join(str(path) for path in stale_paths[:8])
        raise FormalEvalConfigurationError(
            "formal runs must start with empty dynamic/action memory and no prior "
            f"attempt lineage; use the next untouched run directory. Found: {rendered}"
        )


def validate_formal_freeze_commit(
    args: Any,
    *,
    repo_root: Path,
    codebase_snapshot: Mapping[str, Any],
) -> None:
    """Require every formal run to execute from the tagged freeze commit."""

    if not str(getattr(args, "eval_protocol", "")).strip():
        return
    head = str(codebase_snapshot.get("git_head", "")).strip() or _git_revision(
        Path(repo_root),
        "HEAD",
    )
    tagged_commit = _git_revision(
        Path(repo_root),
        f"refs/tags/{FORMAL_EVAL_GIT_TAG}^{{commit}}",
    )
    if not tagged_commit:
        raise FormalEvalConfigurationError(
            f"formal evaluation requires git tag {FORMAL_EVAL_GIT_TAG!r}"
        )
    if head != tagged_commit:
        raise FormalEvalConfigurationError(
            "formal evaluation HEAD must equal the frozen tag commit: "
            f"HEAD={head or '<missing>'}, {FORMAL_EVAL_GIT_TAG}={tagged_commit}"
        )


def validate_formal_initial_runtime(args: Any, state: Mapping[str, Any]) -> None:
    """Reject a Term1 scene that does not match the frozen physical setup."""

    if not str(getattr(args, "eval_protocol", "")).strip():
        return
    payload = state.get("payload", {})
    payload = payload if isinstance(payload, Mapping) else {}
    upper_body = state.get("upper_body_pose_lock", {})
    upper_body = upper_body if isinstance(upper_body, Mapping) else {}
    target = state.get("target", {})
    target = target if isinstance(target, Mapping) else {}
    failures: list[str] = []

    _check_close(
        failures,
        "payload.mass_kg",
        payload.get("mass_kg"),
        FORMAL_PAYLOAD_MASS_KG,
    )
    _check_close(
        failures,
        "payload.rearward_offset_m",
        payload.get("rearward_offset_m"),
        FORMAL_PAYLOAD_REARWARD_OFFSET_M,
    )
    pose_name = str(upper_body.get("pose_name", "")).strip()
    target_pose_name = str(upper_body.get("target_pose_name", "")).strip()
    if (
        pose_name != FORMAL_UPPER_BODY_POSE
        or target_pose_name != FORMAL_UPPER_BODY_POSE
    ):
        failures.append(
            "upper_body_pose_lock must be "
            f"{FORMAL_UPPER_BODY_POSE!r}; got pose={pose_name!r}, "
            f"target={target_pose_name!r}"
        )
    _check_close(
        failures, "upper_body_pose_lock.kp", upper_body.get("kp"), FORMAL_UPPER_BODY_KP
    )
    _check_close(
        failures, "upper_body_pose_lock.kd", upper_body.get("kd"), FORMAL_UPPER_BODY_KD
    )
    _check_close(
        failures,
        "upper_body_pose_lock.configured_ramp_s",
        upper_body.get("configured_ramp_s"),
        FORMAL_UPPER_BODY_RAMP_S,
    )
    _check_close(
        failures,
        "upper_body_pose_lock.palm_tilt_deg",
        upper_body.get("palm_tilt_deg"),
        FORMAL_PALM_TILT_DEG,
    )
    solver = str(state.get("mujoco_solver", "")).strip().lower()
    if solver != FORMAL_MUJOCO_SOLVER:
        failures.append(
            f"mujoco_solver expected {FORMAL_MUJOCO_SOLVER!r}, got {solver!r}"
        )
    for key, expected in (
        ("center_x_m", FORMAL_TARGET_CENTER_X_M),
        ("center_y_m", FORMAL_TARGET_CENTER_Y_M),
        ("center_z_m", FORMAL_TARGET_CENTER_Z_M),
        ("radius_m", FORMAL_TARGET_RADIUS_M),
        ("half_height_m", FORMAL_TARGET_HALF_HEIGHT_M),
    ):
        _check_close(failures, f"target.{key}", target.get(key), expected)
    if failures:
        raise FormalEvalConfigurationError(
            "formal Term1 runtime does not match the frozen physical setup: "
            + "; ".join(failures)
        )


def _check_close(
    failures: list[str],
    name: str,
    actual: object,
    expected: float,
    *,
    tolerance: float = 1.0e-6,
) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        failures.append(f"{name} is missing")
        return
    if abs(value - expected) > tolerance:
        failures.append(f"{name} expected {expected}, got {value}")


def formal_task_overrides(eval_protocol: str) -> dict[str, Any]:
    if not str(eval_protocol).strip():
        return {}
    return {
        "target_xy_tolerance_m": FORMAL_TARGET_XY_TOLERANCE_M,
        "target_z_tolerance_m": FORMAL_TARGET_Z_TOLERANCE_M,
        "max_terminal_payload_tilt_deg": (FORMAL_MAX_TERMINAL_PAYLOAD_TILT_DEG),
        "require_terminal_payload_tilt_measurement": True,
        "robot_exit_min_distance_from_target_m": (FORMAL_ROBOT_EXIT_MIN_DISTANCE_M),
    }


def formal_allows_ancestor_rewind(eval_protocol: str) -> bool:
    """Formal runs only roll failed actions back to their latest checkpoint."""

    return not bool(str(eval_protocol).strip())


def write_formal_run_manifest(
    *,
    args: Any,
    repo_root: Path,
    codebase_snapshot: Mapping[str, Any],
    task: Any,
    status: str,
    model_usage: Mapping[str, Any] | None = None,
) -> Path | None:
    """Write or update the auditable manifest for one formal run."""

    if not str(getattr(args, "eval_protocol", "")).strip():
        return None
    from praxis_coop.cap_policy.static_specs import STATIC_SKILL_CATALOG_PATH

    root = Path(repo_root)
    output_dir = Path(args.output_dir)
    model = formal_backbone(args.planner_model)
    resolved_model = _resolved_model(model_usage)
    manifest_path = output_dir / "run_manifest.json"
    existing_manifest = _read_json_object(manifest_path)
    source_files = codebase_snapshot.get("files", {})
    source_files = source_files if isinstance(source_files, dict) else {}
    verifier_config = {
        "physical_safety_profile": args.physical_safety_profile,
        "target_xy_tolerance_m": task.target_xy_tolerance_m,
        "target_z_tolerance_m": task.target_z_tolerance_m,
        "max_terminal_payload_tilt_deg": task.max_terminal_payload_tilt_deg,
        "require_terminal_payload_tilt_measurement": (
            task.require_terminal_payload_tilt_measurement
        ),
        "robot_exit_min_distance_from_target_m": (
            task.robot_exit_min_distance_from_target_m
        ),
        "require_robots_exit": task.require_robots_exit,
        "terminal_stable_s": FORMAL_TERMINAL_STABLE_S,
    }
    effective_arguments = {
        key: _json_value(value)
        for key, value in sorted(vars(args).items())
        if "api_key" not in key
    }
    configured_route_spec = getattr(args, "route_spec", None)
    route_spec_path = (
        Path(configured_route_spec)
        if configured_route_spec
        else root / FORMAL_ROUTE_SPEC_RELATIVE_PATH
    )
    route_spec_sha256 = _sha256_file(route_spec_path)
    current_memory_sha256 = _sha256_paths(formal_memory_paths(output_dir).values())
    existing_artifact_hashes = existing_manifest.get("artifact_hashes", {})
    existing_artifact_hashes = (
        existing_artifact_hashes
        if isinstance(existing_artifact_hashes, Mapping)
        else {}
    )
    initial_memory_sha256 = (
        str(existing_artifact_hashes.get("initial_online_memory_sha256", "")).strip()
        or current_memory_sha256
    )
    manifest: dict[str, Any] = {
        "kind": "coevolve_formal_run_manifest",
        "schema_version": 1,
        "protocol": asdict(PROTOCOL),
        "status": str(status),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": FORMAL_BENCHMARK_NAME,
        "route_spec_sha256": route_spec_sha256,
        "run_index": int(args.run_index),
        "seed": int(args.seed),
        "model": {
            "requested": model.requested_model,
            "provider": model.provider,
            "planner_backend": model.planner_backend,
            "resolved": resolved_model,
            "identity_verified": resolved_model == model.requested_model,
        },
        "codebase": {
            "git_head": codebase_snapshot.get("git_head", ""),
            "git_branch": codebase_snapshot.get("git_branch", ""),
            "source_sha256": codebase_snapshot.get("source_sha256", ""),
            "dirty_entries": codebase_snapshot.get("git_dirty_entries", []),
        },
        "artifact_hashes": {
            "prompt_sha256": source_files.get(
                "src/praxis_coop/cap_policy/prompts.py", ""
            ),
            "skill_api_sha256": _combined_source_hash(
                source_files,
                (
                    "src/praxis_coop/cap_policy/skill_api.py",
                    "src/praxis_coop/cap_policy/static_specs.py",
                    "src/praxis_coop/cap_policy/coordination_specs.py",
                ),
            ),
            "verifier_source_sha256": _combined_source_hash(
                source_files,
                (
                    "src/praxis_coop/cap_policy/evaluator.py",
                    "src/praxis_coop/cap_policy/simulation_diagnostics.py",
                    "src/praxis_coop/cap_policy/route_spec.py",
                    "src/praxis_coop/cap_policy/transition_verifier.py",
                    "src/praxis_coop/sonic/safety.py",
                ),
            ),
            "route_spec_sha256": route_spec_sha256,
            "verifier_config_sha256": _json_sha256(verifier_config),
            "scene_sha256": _sha256_file(
                root / "assets/manual_scenes/two_g1_box_push_layout.usda"
            ),
            # Hash the catalog actually loaded by this runtime, not private output.
            # Filename-keyed hashing preserves the historical single-file digest.
            "static_memory_sha256": _sha256_paths((STATIC_SKILL_CATALOG_PATH,)),
            "formal_arguments_sha256": _json_sha256(FORMAL_ARGUMENT_OVERRIDES),
            "initial_online_memory_sha256": initial_memory_sha256,
            "final_online_memory_sha256": current_memory_sha256,
        },
        "result_artifact_hashes": {
            "billing_reconciliation_sha256": _sha256_file(
                output_dir / "billing_reconciliation.json"
            ),
            "billing_post_reconciliation_sha256": _sha256_file(
                output_dir / "billing_post_reconciliation.json"
            ),
            "api_call_costs_json_sha256": _sha256_file(
                output_dir / "api_call_costs.json"
            ),
            "api_call_costs_csv_sha256": _sha256_file(
                output_dir / "api_call_costs.csv"
            ),
            "model_usage_sha256": _sha256_file(output_dir / "model_usage.json"),
            "metrics_sha256": _sha256_file(output_dir / "metrics.json"),
            "final_summary_sha256": _sha256_file(
                output_dir / "final_summary.json"
            ),
        },
        "verifier_config": verifier_config,
        "memory": {
            "condition": "empty_at_run_start_then_run_local_online_updates",
            "cross_run_retrieval": False,
            "paths": {
                key: str(path) for key, path in formal_memory_paths(output_dir).items()
            },
        },
        "budgets": {
            "physical_transactions": FORMAL_PHYSICAL_BUDGET,
            "model_attempts": FORMAL_MODEL_ATTEMPT_BUDGET,
        },
        "runtime": {
            "mujoco_version": _package_version("mujoco"),
            "sonic_git_head": _git_head(root / "third_party/GR00T-WholeBodyControl"),
            "sonic_runtime_artifacts": {
                relative: _sha256_file(root / relative)
                for relative in SONIC_RUNTIME_ARTIFACTS
            },
            "sonic_deploy_source_sha256": _sha256_tree(
                root / "third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src"
            ),
            "term1_summary_sha256": _sha256_file(output_dir / "term1_summary.json"),
        },
        "effective_arguments": effective_arguments,
        "model_usage": _json_value(model_usage or {}),
    }
    fingerprint_runtime = dict(manifest["runtime"])
    fingerprint_runtime.pop("term1_summary_sha256", None)
    fingerprint_artifacts = dict(manifest["artifact_hashes"])
    fingerprint_artifacts.pop("final_online_memory_sha256", None)
    manifest["evaluation_fingerprint_sha256"] = _json_sha256(
        {
            "protocol": manifest["protocol"],
            "formal_arguments": FORMAL_ARGUMENT_OVERRIDES,
            "source_sha256": manifest["codebase"]["source_sha256"],
            "artifact_hashes": fingerprint_artifacts,
            "runtime": fingerprint_runtime,
        }
    )
    manifest["manifest_sha256"] = _json_sha256(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _resolved_model(model_usage: Mapping[str, Any] | None) -> str:
    if not isinstance(model_usage, Mapping):
        return ""
    for bucket_name in ("text", "vision"):
        bucket = model_usage.get(bucket_name, {})
        if not isinstance(bucket, Mapping):
            continue
        calls = bucket.get("calls", ())
        if not isinstance(calls, (list, tuple)):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            resolved = str(call.get("resolved_model", "")).strip()
            if resolved:
                return resolved
    return ""


def _combined_source_hash(
    source_files: Mapping[str, Any], paths: tuple[str, ...]
) -> str:
    return _json_sha256({path: str(source_files.get(path, "")) for path in paths})


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _sha256_tree(path: Path) -> str:
    files = (
        {
            item.relative_to(path).as_posix(): _sha256_file(item)
            for item in sorted(path.rglob("*"))
            if item.is_file()
        }
        if path.is_dir()
        else {}
    )
    return _json_sha256(files)


def _sha256_paths(paths: Any) -> str:
    files = {
        Path(path).name: _sha256_file(Path(path))
        for path in sorted((Path(path) for path in paths), key=lambda item: item.name)
    }
    return _json_sha256(files)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_head(path: Path) -> str:
    if not path.exists():
        return "unavailable"
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _git_revision(path: Path, revision: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--verify", revision],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
