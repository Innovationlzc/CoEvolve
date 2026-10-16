"""Strict, hash-bound wrapper for the model-free appendix replay.

The historical :mod:`scripts.manage_accepted_episode` utility remains
unchanged.  This module puts a closed-world evidence boundary around it: the
successful source result, finalized accepted manifest, exported trajectory,
executor bytes, full action sequence, and controller-history mode are pinned
before execution.  A result is accepted only when the wrapper launched that
exact complete sequence and the produced summary/config reconcile with it.

Replay remains an appendix diagnostic; none of these records is a task run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

from ..serialization import canonical_json, sha256_json, strict_json_loads


STRICT_REPLAY_MANIFEST_KIND = "coevolve_strict_model_free_replay_manifest"
STRICT_REPLAY_MANIFEST_SCHEMA_VERSION = 1
STRICT_REPLAY_RESULT_KIND = "coevolve_strict_model_free_replay_result"
STRICT_REPLAY_RESULT_SCHEMA_VERSION = 1
ACCEPTED_MANIFEST_PIN = "accepted_episode_manifest_sha256"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EXECUTOR = _REPO_ROOT / "scripts" / "manage_accepted_episode.py"
_ACCEPTED_MANIFEST_CANDIDATES = (
    PurePosixPath("memory/v83_native/accepted_episode_manifest.json"),
    PurePosixPath("memory/accepted_episode_manifest.json"),
)


class ReplayHistoryMode(str, Enum):
    CONTINUOUS = "continuous_controller_history"
    RESET_EACH_CHECKPOINT = "reset_each_accepted_checkpoint"


class StrictReplayEvidenceError(ValueError):
    """A replay source or result is not byte- and sequence-auditable."""


@dataclass(frozen=True, slots=True)
class StrictReplayBinding:
    manifest_path: Path
    manifest_sha256: str
    run_root: Path
    package_root: Path
    executor_path: Path
    mode: ReplayHistoryMode
    action_count: int
    action_sequence_sha256: str
    document: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StrictReplayResult:
    result_path: Path
    result_sha256: str
    replay_dir: Path
    status: str
    full_success: bool
    safe_replay_prefix: int
    total_actions: int
    failed_step: int | None
    maximum_payload_trajectory_divergence_m: float | None


def prepare_strict_replay_manifest(
    run_root: str | Path,
    package_root: str | Path,
    *,
    mode: ReplayHistoryMode | str,
    destination: str | Path | None = None,
) -> StrictReplayBinding:
    """Exclusively freeze one complete, model-free replay request."""

    run = _strict_directory(Path(run_root), "source run")
    package = _strict_directory(Path(package_root), "trajectory package")
    _require_contained(package, run, "trajectory package")
    selected_mode = _mode(mode)

    run_manifest_path = run / "run_manifest.json"
    run_manifest, run_manifest_sha = _read_json_artifact(
        run_manifest_path,
        "source run manifest",
    )
    if run_manifest.get("status") != "SUCCESS":
        raise StrictReplayEvidenceError(
            "appendix replay requires a manifest-published SUCCESS source run"
        )
    result_hashes = _mapping(
        run_manifest.get("result_artifact_hashes"),
        "run manifest result_artifact_hashes",
    )
    accepted_sha = _sha(
        result_hashes.get(ACCEPTED_MANIFEST_PIN),
        f"run manifest {ACCEPTED_MANIFEST_PIN}",
    )
    accepted_path, accepted, actual_accepted_sha = _resolve_accepted_manifest(
        run,
        accepted_sha,
    )
    if actual_accepted_sha != accepted_sha:  # defensive; resolver already checks
        raise StrictReplayEvidenceError("accepted manifest pin changed")

    trajectory_path = package / "trajectory_manifest.json"
    trajectory, trajectory_sha = _read_json_artifact(
        trajectory_path,
        "trajectory manifest",
    )
    action_sequence = _validated_action_sequence(accepted, trajectory)
    action_count = len(action_sequence)
    executor_path = _strict_file(_DEFAULT_EXECUTOR, "replay executor")
    executor_sha = _sha256_file(executor_path)
    run_id = _run_id(run_manifest)

    document = {
        "kind": STRICT_REPLAY_MANIFEST_KIND,
        "schema_version": STRICT_REPLAY_MANIFEST_SCHEMA_VERSION,
        "diagnostic_only": True,
        "run_id": run_id,
        "history_mode": selected_mode.value,
        "source_run_manifest": _ref(run_manifest_path, run, run_manifest_sha),
        "accepted_episode_manifest": _ref(
            accepted_path,
            run,
            accepted_sha,
        ),
        "trajectory_manifest": _ref(trajectory_path, run, trajectory_sha),
        "executor": {
            "relative_path": executor_path.relative_to(_REPO_ROOT).as_posix(),
            "sha256": executor_sha,
        },
        "execution_contract": {
            "execute": True,
            "from_step": 1,
            "through_step": action_count,
            "gate_timeout_s": 10.0,
            "planner_reset_each_step": (
                selected_mode is ReplayHistoryMode.RESET_EACH_CHECKPOINT
            ),
            "model_calls_allowed": False,
        },
        "action_count": action_count,
        "action_sequence_sha256": sha256_json(action_sequence),
    }
    target = (
        Path(destination)
        if destination is not None
        else package / f"strict_replay_manifest_{selected_mode.value}.json"
    )
    target = _normalized_absolute(target, "strict replay manifest destination")
    _require_contained(target, package, "strict replay manifest destination")
    _exclusive_json(target, document)
    return load_strict_replay_manifest(target, run_root=run, package_root=package)


def load_strict_replay_manifest(
    manifest_path: str | Path,
    *,
    run_root: str | Path,
    package_root: str | Path,
) -> StrictReplayBinding:
    """Reload all pinned bytes and reconstruct the action sequence."""

    run = _strict_directory(Path(run_root), "source run")
    package = _strict_directory(Path(package_root), "trajectory package")
    _require_contained(package, run, "trajectory package")
    path = _strict_file(Path(manifest_path), "strict replay manifest")
    _require_contained(path, package, "strict replay manifest")
    value, manifest_sha = _read_json_artifact(path, "strict replay manifest")
    _exact_fields(
        value,
        {
            "kind",
            "schema_version",
            "diagnostic_only",
            "run_id",
            "history_mode",
            "source_run_manifest",
            "accepted_episode_manifest",
            "trajectory_manifest",
            "executor",
            "execution_contract",
            "action_count",
            "action_sequence_sha256",
        },
        "strict replay manifest",
    )
    if (
        value.get("kind") != STRICT_REPLAY_MANIFEST_KIND
        or value.get("schema_version") != STRICT_REPLAY_MANIFEST_SCHEMA_VERSION
        or value.get("diagnostic_only") is not True
    ):
        raise StrictReplayEvidenceError("strict replay manifest identity is invalid")
    selected_mode = _mode(value.get("history_mode"))

    run_manifest = _load_ref(run, value.get("source_run_manifest"), "run manifest")
    if run_manifest.get("status") != "SUCCESS":
        raise StrictReplayEvidenceError("replay source run is not SUCCESS")
    if _run_id(run_manifest) != value.get("run_id"):
        raise StrictReplayEvidenceError("replay run_id differs from source manifest")
    result_hashes = _mapping(
        run_manifest.get("result_artifact_hashes"),
        "run manifest result_artifact_hashes",
    )
    accepted_ref = _mapping(
        value.get("accepted_episode_manifest"),
        "accepted manifest reference",
    )
    if accepted_ref.get("sha256") != result_hashes.get(ACCEPTED_MANIFEST_PIN):
        raise StrictReplayEvidenceError(
            "accepted manifest is not pinned by the source run manifest"
        )
    accepted = _load_ref(run, accepted_ref, "accepted manifest")
    trajectory = _load_ref(run, value.get("trajectory_manifest"), "trajectory")
    action_sequence = _validated_action_sequence(accepted, trajectory)
    action_count = len(action_sequence)
    if value.get("action_count") != action_count:
        raise StrictReplayEvidenceError("replay action_count is not derived")
    sequence_sha = sha256_json(action_sequence)
    if value.get("action_sequence_sha256") != sequence_sha:
        raise StrictReplayEvidenceError("replay action-sequence hash differs")

    executor = _mapping(value.get("executor"), "replay executor")
    _exact_fields(executor, {"relative_path", "sha256"}, "replay executor")
    if executor.get("relative_path") != "scripts/manage_accepted_episode.py":
        raise StrictReplayEvidenceError("replay executor path is not canonical")
    executor_path = _strict_file(
        _REPO_ROOT / str(executor["relative_path"]),
        "replay executor",
    )
    if _sha256_file(executor_path) != _sha(executor.get("sha256"), "executor hash"):
        raise StrictReplayEvidenceError("replay executor bytes changed")
    _validate_execution_contract(
        value.get("execution_contract"),
        selected_mode,
        action_count,
    )
    return StrictReplayBinding(
        manifest_path=path,
        manifest_sha256=manifest_sha,
        run_root=run,
        package_root=package,
        executor_path=executor_path,
        mode=selected_mode,
        action_count=action_count,
        action_sequence_sha256=sequence_sha,
        document=value,
    )


def execute_strict_replay(
    binding: StrictReplayBinding,
    *,
    python_executable: str | Path = sys.executable,
    timeout_s: float | None = None,
) -> StrictReplayResult:
    """Run the byte-pinned executor once and publish a derived result receipt."""

    if not isinstance(binding, StrictReplayBinding):
        raise StrictReplayEvidenceError("binding must be StrictReplayBinding")
    # Revalidate immediately before dispatch; a stale in-memory binding is not
    # authorization to execute changed source bytes.
    binding = load_strict_replay_manifest(
        binding.manifest_path,
        run_root=binding.run_root,
        package_root=binding.package_root,
    )
    # Virtual environments normally expose ``bin/python`` as a symlink.  The
    # interpreter is a launch tool rather than replay evidence, so resolve
    # that symlink to one real executable while continuing to reject symlinks
    # for every hash-bound evidence artifact.
    interpreter = _resolved_executable_file(
        Path(python_executable),
        "Python interpreter",
    )
    replay_root = binding.package_root / "replays"
    before = _replay_directories(replay_root)
    contract = _mapping(
        binding.document["execution_contract"],
        "execution contract",
    )
    argv = [
        str(interpreter),
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
        str(float(contract["gate_timeout_s"])),
        (
            "--planner-reset-each-step"
            if binding.mode is ReplayHistoryMode.RESET_EACH_CHECKPOINT
            else "--no-planner-reset-each-step"
        ),
    ]
    completed = subprocess.run(
        argv,
        cwd=_REPO_ROOT,
        check=False,
        timeout=timeout_s,
    )
    after = _replay_directories(replay_root)
    created = sorted(after - before)
    if len(created) != 1:
        raise StrictReplayEvidenceError(
            "strict replay must create exactly one new replay directory"
        )
    return validate_and_publish_strict_replay_result(
        binding,
        created[0],
        process_returncode=completed.returncode,
        argv=argv,
    )


def validate_and_publish_strict_replay_result(
    binding: StrictReplayBinding,
    replay_dir: str | Path,
    *,
    process_returncode: int,
    argv: Sequence[str],
) -> StrictReplayResult:
    """Validate one executor output and write its immutable derived receipt."""

    binding = load_strict_replay_manifest(
        binding.manifest_path,
        run_root=binding.run_root,
        package_root=binding.package_root,
    )
    replay = _strict_directory(Path(replay_dir), "replay output")
    _require_contained(replay, binding.package_root / "replays", "replay output")
    config_path = replay / "replay_config.json"
    summary_path = replay / "replay_summary.json"
    config, config_sha = _read_json_artifact(config_path, "replay config")
    summary, summary_sha = _read_json_artifact(summary_path, "replay summary")
    _validate_replay_config(config, binding)
    derived = _derive_replay_summary(summary, binding.action_count)
    if not isinstance(process_returncode, int) or isinstance(process_returncode, bool):
        raise StrictReplayEvidenceError("replay process return code is invalid")
    expected_codes = {0} if derived["full_success"] else {2, 3}
    if process_returncode not in expected_codes:
        raise StrictReplayEvidenceError(
            "replay process return code contradicts its physical summary"
        )
    expected_argv = _expected_argv_suffix(binding)
    rendered_argv = [str(item) for item in argv]
    if rendered_argv[1:] != expected_argv:
        raise StrictReplayEvidenceError("replay invocation differs from frozen contract")

    document = {
        "kind": STRICT_REPLAY_RESULT_KIND,
        "schema_version": STRICT_REPLAY_RESULT_SCHEMA_VERSION,
        "diagnostic_only": True,
        "strict_replay_manifest_sha256": binding.manifest_sha256,
        "history_mode": binding.mode.value,
        "process_returncode": process_returncode,
        "replay_config": _ref(config_path, binding.run_root, config_sha),
        "replay_summary": _ref(summary_path, binding.run_root, summary_sha),
        **derived,
    }
    result_path = (
        binding.package_root
        / f"strict_replay_result_{binding.mode.value}.json"
    )
    _exclusive_json(result_path, document)
    result_sha = _sha256_file(result_path)
    return StrictReplayResult(
        result_path=result_path,
        result_sha256=result_sha,
        replay_dir=replay,
        status=str(derived["status"]),
        full_success=bool(derived["full_success"]),
        safe_replay_prefix=int(derived["safe_replay_prefix"]),
        total_actions=int(derived["total_actions"]),
        failed_step=derived["failed_step"],
        maximum_payload_trajectory_divergence_m=derived[
            "maximum_payload_trajectory_divergence_m"
        ],
    )


def _validated_action_sequence(
    accepted: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    accepted_steps = accepted.get("steps")
    trajectory_steps = trajectory.get("steps")
    if not isinstance(accepted_steps, list) or not accepted_steps:
        raise StrictReplayEvidenceError("accepted manifest has no steps")
    if not isinstance(trajectory_steps, list) or len(trajectory_steps) != len(
        accepted_steps
    ):
        raise StrictReplayEvidenceError(
            "trajectory does not contain the complete accepted sequence"
        )
    if trajectory.get("accepted_action_count") != len(accepted_steps):
        raise StrictReplayEvidenceError("trajectory accepted_action_count differs")
    sequence: list[dict[str, Any]] = []
    previous_after = accepted.get("initial_checkpoint")
    for index, (source, exported) in enumerate(
        zip(accepted_steps, trajectory_steps),
        1,
    ):
        source = _mapping(source, f"accepted step {index}")
        exported = _mapping(exported, f"trajectory step {index}")
        expected_fields = (
            "attempt",
            "script_id",
            "checkpoint_before",
            "checkpoint_after",
            "transition_outcome",
        )
        if exported.get("sequence_index") != index:
            raise StrictReplayEvidenceError("trajectory sequence indices are not contiguous")
        for field in expected_fields:
            if exported.get(field) != source.get(field):
                raise StrictReplayEvidenceError(
                    f"trajectory step {index} {field} differs from accepted manifest"
                )
        if source.get("checkpoint_before") != previous_after:
            raise StrictReplayEvidenceError("accepted checkpoint sequence is discontinuous")
        previous_after = source.get("checkpoint_after")
        source_scripts = _mapping(source.get("robot_scripts"), "accepted robot scripts")
        exported_scripts = _mapping(
            exported.get("robot_scripts"),
            "trajectory robot scripts",
        )
        code_hashes: dict[str, str] = {}
        for robot_id in ("front", "back"):
            source_script = _mapping(
                source_scripts.get(robot_id),
                f"accepted {robot_id} script",
            )
            exported_script = _mapping(
                exported_scripts.get(robot_id),
                f"trajectory {robot_id} script",
            )
            source_code = _text(source_script.get("code"), "accepted policy code")
            exported_code = _text(exported_script.get("code"), "exported policy code")
            if exported_code.rstrip() != source_code.rstrip():
                raise StrictReplayEvidenceError(
                    f"trajectory step {index} {robot_id} code differs"
                )
            code_hashes[robot_id] = hashlib.sha256(
                source_code.rstrip().encode("utf-8")
            ).hexdigest()
        sequence.append(
            {
                "sequence_index": index,
                **{field: source.get(field) for field in expected_fields},
                "robot_code_sha256": code_hashes,
            }
        )
    if previous_after != accepted.get("latest_checkpoint"):
        raise StrictReplayEvidenceError("accepted latest checkpoint differs")
    return sequence


def _derive_replay_summary(
    summary: Mapping[str, Any],
    total_actions: int,
) -> dict[str, Any]:
    results = summary.get("results")
    if not isinstance(results, list) or not results:
        raise StrictReplayEvidenceError("replay summary has no per-step results")
    safe_prefix = 0
    first_failed: int | None = None
    divergence: list[float] = []
    for expected_index, raw in enumerate(results, 1):
        row = _mapping(raw, f"replay result {expected_index}")
        if row.get("sequence_index") != expected_index:
            raise StrictReplayEvidenceError("replay result sequence is not contiguous")
        success = row.get("physical_success")
        if not isinstance(success, bool):
            raise StrictReplayEvidenceError("replay physical_success must be boolean")
        if success and first_failed is None:
            safe_prefix += 1
        elif not success and first_failed is None:
            first_failed = expected_index
        elif success:
            raise StrictReplayEvidenceError("replay continued after a failed physical step")
        deviation = row.get("accepted_endpoint_deviation")
        if success and isinstance(deviation, Mapping):
            xy = _optional_number(deviation.get("payload_xy_error_m"))
            z = _optional_number(deviation.get("payload_z_error_m"))
            if xy is not None and z is not None:
                divergence.append(math.hypot(xy, z))
    status = _text(summary.get("status"), "replay status")
    if status not in {
        "completed",
        "completed_with_final_gate_failure",
        "failed_and_rolled_back",
        "checkpoint_roundtrip_failed_and_rolled_back",
    }:
        raise StrictReplayEvidenceError("replay status is not recognized")
    final_gate = summary.get("final_physical_gate")
    final_gate_success = (
        final_gate.get("success")
        if isinstance(final_gate, Mapping)
        else None
    )
    full_success = (
        status == "completed"
        and first_failed is None
        and len(results) == total_actions
        and safe_prefix == total_actions
        and final_gate_success is True
    )
    if len(results) > total_actions:
        raise StrictReplayEvidenceError("replay contains more steps than the source")
    summary_failed = summary.get("failed_step")
    if full_success:
        if summary_failed is not None:
            raise StrictReplayEvidenceError("successful replay declares a failed step")
    elif first_failed is not None:
        if summary_failed != first_failed:
            raise StrictReplayEvidenceError("replay failed_step is not derived")
        if summary.get("completed_steps") != safe_prefix:
            raise StrictReplayEvidenceError("replay completed_steps is not derived")
    elif not (
        status == "completed_with_final_gate_failure"
        and len(results) == total_actions
        and safe_prefix == total_actions
        and summary_failed is None
        and final_gate_success is False
    ):
        raise StrictReplayEvidenceError(
            "non-success replay lacks a derived step or final-gate failure"
        )
    return {
        "status": status,
        "full_success": full_success,
        "safe_replay_prefix": safe_prefix,
        "total_actions": total_actions,
        "failed_step": first_failed,
        "maximum_payload_trajectory_divergence_m": (
            max(divergence) if divergence else None
        ),
    }


def _validate_replay_config(
    config: Mapping[str, Any],
    binding: StrictReplayBinding,
) -> None:
    expected = {
        "from_step": 1,
        "through_step": binding.action_count,
        "steps_requested": binding.action_count,
        "planner_reset_each_step": (
            binding.mode is ReplayHistoryMode.RESET_EACH_CHECKPOINT
        ),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise StrictReplayEvidenceError(f"replay config {key} differs")


def _validate_execution_contract(
    value: object,
    mode: ReplayHistoryMode,
    action_count: int,
) -> None:
    contract = _mapping(value, "execution contract")
    expected = {
        "execute": True,
        "from_step": 1,
        "through_step": action_count,
        "gate_timeout_s": 10.0,
        "planner_reset_each_step": mode is ReplayHistoryMode.RESET_EACH_CHECKPOINT,
        "model_calls_allowed": False,
    }
    if dict(contract) != expected:
        raise StrictReplayEvidenceError("strict replay execution contract differs")


def _expected_argv_suffix(binding: StrictReplayBinding) -> list[str]:
    return [
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
        (
            "--planner-reset-each-step"
            if binding.mode is ReplayHistoryMode.RESET_EACH_CHECKPOINT
            else "--no-planner-reset-each-step"
        ),
    ]


def _resolve_accepted_manifest(
    run: Path,
    expected_sha: str,
) -> tuple[Path, Mapping[str, Any], str]:
    matches: list[tuple[Path, Mapping[str, Any], str]] = []
    for relpath in _ACCEPTED_MANIFEST_CANDIDATES:
        candidate = run.joinpath(*relpath.parts)
        if not candidate.is_file() or candidate.is_symlink():
            continue
        value, digest = _read_json_artifact(candidate, "accepted manifest")
        if digest == expected_sha:
            matches.append((candidate, value, digest))
    if len(matches) != 1:
        raise StrictReplayEvidenceError(
            "run manifest accepted-episode pin does not resolve exactly once"
        )
    return matches[0]


def _load_ref(root: Path, value: object, label: str) -> Mapping[str, Any]:
    ref = _mapping(value, f"{label} reference")
    _exact_fields(ref, {"relative_path", "sha256", "size_bytes"}, f"{label} reference")
    relative = ref.get("relative_path")
    if not isinstance(relative, str):
        raise StrictReplayEvidenceError(f"{label} path must be a string")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise StrictReplayEvidenceError(f"{label} path is unsafe")
    path = _strict_file(root.joinpath(*pure.parts), label)
    _require_contained(path, root, label)
    raw = path.read_bytes()
    if len(raw) != ref.get("size_bytes"):
        raise StrictReplayEvidenceError(f"{label} size changed")
    if hashlib.sha256(raw).hexdigest() != _sha(ref.get("sha256"), f"{label} hash"):
        raise StrictReplayEvidenceError(f"{label} hash changed")
    try:
        payload = strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StrictReplayEvidenceError(f"{label} is not strict JSON") from exc
    return _mapping(payload, label)


def _ref(path: Path, root: Path, digest: str) -> dict[str, Any]:
    _require_contained(path, root, "artifact reference")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise StrictReplayEvidenceError("artifact changed while building replay manifest")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "size_bytes": len(raw),
    }


def _run_id(manifest: Mapping[str, Any]) -> str:
    experiment = manifest.get("experiment")
    run_id = experiment.get("run_id") if isinstance(experiment, Mapping) else None
    if run_id is None:
        run_id = manifest.get("run_id")
    return _text(run_id, "source run_id")


def _mode(value: ReplayHistoryMode | str | object) -> ReplayHistoryMode:
    try:
        return value if isinstance(value, ReplayHistoryMode) else ReplayHistoryMode(value)
    except (TypeError, ValueError) as exc:
        raise StrictReplayEvidenceError("unsupported replay history mode") from exc


def _replay_directories(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    if root.is_symlink() or not root.is_dir():
        raise StrictReplayEvidenceError("replays root must be a real directory")
    return {
        item.resolve()
        for item in root.iterdir()
        if item.is_dir() and not item.is_symlink() and item.name.startswith("replay_")
    }


def _read_json_artifact(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    path = _strict_file(path, label)
    raw = path.read_bytes()
    try:
        payload = strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StrictReplayEvidenceError(f"{label} is not strict JSON") from exc
    return _mapping(payload, label), hashlib.sha256(raw).hexdigest()


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical_json(value) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise StrictReplayEvidenceError(f"refusing to overwrite {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _strict_file(path: Path, label: str) -> Path:
    value = _normalized_absolute(path, label)
    if value.is_symlink() or not value.is_file():
        raise StrictReplayEvidenceError(f"{label} must be a regular file")
    return value


def _resolved_executable_file(path: Path, label: str) -> Path:
    try:
        value = path.resolve(strict=True)
    except OSError as exc:
        raise StrictReplayEvidenceError(f"{label} is unavailable") from exc
    if value.is_symlink() or not value.is_file():
        raise StrictReplayEvidenceError(f"{label} must resolve to a regular file")
    if not os.access(value, os.X_OK):
        raise StrictReplayEvidenceError(f"{label} must be executable")
    return value


def _strict_directory(path: Path, label: str) -> Path:
    value = _normalized_absolute(path, label)
    if value.is_symlink() or not value.is_dir():
        raise StrictReplayEvidenceError(f"{label} must be a real directory")
    return value


def _normalized_absolute(path: Path, label: str) -> Path:
    value = path if path.is_absolute() else path.resolve()
    if value != Path(os.path.normpath(str(value))):
        raise StrictReplayEvidenceError(f"{label} must be normalized")
    return value


def _require_contained(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StrictReplayEvidenceError(f"{label} is outside its allowed root") from exc


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictReplayEvidenceError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise StrictReplayEvidenceError(f"{label} fields differ from schema")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrictReplayEvidenceError(f"{label} must be non-empty text")
    return value.strip()


def _sha(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise StrictReplayEvidenceError(f"{label} must be lowercase SHA-256")
    return text


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictReplayEvidenceError("replay deviation must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise StrictReplayEvidenceError("replay deviation must be finite and nonnegative")
    return result


__all__ = [
    "ACCEPTED_MANIFEST_PIN",
    "ReplayHistoryMode",
    "STRICT_REPLAY_MANIFEST_KIND",
    "STRICT_REPLAY_MANIFEST_SCHEMA_VERSION",
    "STRICT_REPLAY_RESULT_KIND",
    "STRICT_REPLAY_RESULT_SCHEMA_VERSION",
    "StrictReplayBinding",
    "StrictReplayEvidenceError",
    "StrictReplayResult",
    "execute_strict_replay",
    "load_strict_replay_manifest",
    "prepare_strict_replay_manifest",
    "validate_and_publish_strict_replay_result",
]
