"""Independent, hash-bound validation of formal terminal outcomes.

The collector never trusts a runner-authored success boolean.  A schema-v3
receipt pins an immutable MuJoCo sample trace; placement, height, tilt,
contacts, robot pose/clearance, and stability are recomputed from those raw
samples.  The exact compiled model and realized scene are also bound to the
suite-plan scene evidence.  SYSTEM_STALL and five-drop terminals require
schema-v3 ledgers that are independently reconstructed from immutable raw
checkpoint/cycle or per-transaction artifacts; ledger prose is never treated
as proof of either terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .evidence_artifacts import BoundArtifactError, BoundJsonArtifact, load_bound_json_artifact
from .terminal_aux_evidence import (
    PAYLOAD_DROP_LEDGER_KIND,
    PAYLOAD_DROP_LEDGER_SCHEMA_VERSION,
    PAYLOAD_DROP_LIMIT,
    SYSTEM_STALL_GENERATION_CYCLES,
    SYSTEM_STALL_LEDGER_KIND,
    SYSTEM_STALL_LEDGER_SCHEMA_VERSION,
    TRUSTED_CHECKPOINT_MIN_PELVIS_HEIGHT_M,
    capability_bound_auxiliary_producer_allowlist,
    recompute_payload_drop_ledger,
    recompute_system_stall_ledger,
)


TERMINAL_EVIDENCE_FILENAME = "terminal_verifier_evidence.json"
TERMINAL_EVIDENCE_KIND = "coevolve_terminal_verifier_evidence"
TERMINAL_EVIDENCE_SCHEMA_VERSION = 3
MUJOCO_TERMINAL_STATE_KIND = "coevolve_mujoco_terminal_state"
MUJOCO_TERMINAL_STATE_SCHEMA_VERSION = 3
MUJOCO_TERMINAL_SAMPLE_TRACE_KIND = "coevolve_mujoco_terminal_sample_trace"
MUJOCO_TERMINAL_SAMPLE_TRACE_SCHEMA_VERSION = 2

# Exact frozen-V83 geometry/predicate.  Any change requires a new terminal
# fingerprint and evidence schema rather than a tolerance edited in place.
TARGET_SUPPORT_RADIUS_M = 0.30
TARGET_SCALE_XYZ_M = (0.60, 0.60, 0.60)
PAYLOAD_HALF_HEIGHT_M = 0.05
TARGET_Z_TOLERANCE_M = 0.08
MAX_TERMINAL_PAYLOAD_TILT_DEG = 20.0
ROBOT_EXIT_MIN_DISTANCE_M = 1.10
UPRIGHT_MIN_PELVIS_HEIGHT_M = 0.68
MAX_UPRIGHT_ROBOT_TILT_DEG = 45.0
TERMINAL_STABLE_WINDOW_S = 3.0
CONTACT_DISTANCE_TOLERANCE_M = 1.0e-6

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_TERMINALS = frozenset({
    "SUCCESS", "SYSTEM_STALL", "PHYSICAL_BUDGET_EXHAUSTED",
    "CONSECUTIVE_PAYLOAD_DROP_LIMIT",
})
_RECEIPT_FIELDS = frozenset({
    "kind", "schema_version", "producer", "run_binding", "terminal_status",
    "physical_transactions", "terminal_state", "terminal_sample_trace",
    "system_stall", "payload_drop",
})
_PRODUCER_FIELDS = frozenset({"id", "source_sha256"})
_RUN_BINDING_FIELDS = frozenset({
    "run_id", "seed", "condition_fingerprint", "terminal_fingerprint",
    "scene_fingerprint", "scene_execution_evidence_fingerprint",
})
_STATE_FIELDS = frozenset({
    "kind", "schema_version", "run_id", "seed", "checkpoint_id",
    "checkpoint_event_id", "simulator",
    "sample_window_s", "sample_count", "payload", "robots", "lifecycle",
})
_SIMULATOR_FIELDS = frozenset({
    "name", "version", "compiled_model_sha256", "scene_realization_sha256",
    "scene_fingerprint", "scene_execution_evidence_fingerprint",
})
_PAYLOAD_FIELDS = frozenset({
    "released", "target_contact_all_samples", "floor_contact_samples",
    "wall_contact_samples", "robot_contact_samples", "xy_error_m", "z_error_m",
    "tilt_deg",
})
_ROBOT_FIELDS = frozenset({"upright", "pelvis_height_m", "distance_to_target_m"})
_LIFECYCLE_FIELDS = frozenset({"release_completed", "retreat_completed", "stand_recovered"})
_TRACE_FIELDS = frozenset({
    "kind", "schema_version", "run_id", "seed", "checkpoint_id",
    "checkpoint_event_id", "simulator",
    "target", "payload_geometry", "sample_window_s", "samples", "lifecycle",
})
_TRACE_TARGET_FIELDS = frozenset({"center_xyz_m", "scale_xyz_m"})
_TRACE_PAYLOAD_GEOMETRY_FIELDS = frozenset({"half_height_m"})
_TRACE_SAMPLE_FIELDS = frozenset({
    "sample_index", "time_s", "payload_released", "payload_pose", "robot_poses",
    "contacts",
})
_POSE_FIELDS = frozenset({"position_xyz_m", "quaternion_wxyz"})
_CONTACT_FIELDS = frozenset({
    "geom1_id", "geom2_id", "geom1_role", "geom2_role", "distance_m",
})
_CONTACT_ROLES = frozenset({
    "payload", "target", "floor", "wall", "front_robot", "back_robot",
})


@dataclass(frozen=True)
class _DerivedPayload:
    released: bool
    target_contact_all_samples: bool
    floor_contact_samples: int
    wall_contact_samples: int
    robot_contact_samples: int
    xy_error_m: float
    z_error_m: float
    tilt_deg: float


@dataclass(frozen=True)
class _DerivedRobot:
    upright: bool
    pelvis_height_m: float
    distance_to_target_m: float


@dataclass(frozen=True)
class _DerivedTrace:
    checkpoint_id: str
    checkpoint_event_id: str
    simulator: Mapping[str, Any]
    sample_window_s: float
    sample_count: int
    payload: _DerivedPayload
    robots: Mapping[str, _DerivedRobot]
    lifecycle: Mapping[str, bool]


@dataclass(frozen=True)
class TerminalEvidenceValidation:
    receipt_sha256: str = ""
    artifact_hashes: Mapping[str, str] | None = None
    stall_model_request_ids: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.invalid_reasons


def validate_terminal_evidence(
    run_root: str | Path,
    *,
    manifest: Mapping[str, Any],
    expected_run_id: str,
    expected_seed: int,
    expected_condition_fingerprint: str,
    expected_terminal_fingerprint: str,
    expected_terminal_status: str,
    expected_scene_fingerprint: str,
    expected_scene_realization_sha256: str,
    expected_compiled_model_sha256: str,
    expected_scene_execution_evidence_fingerprint: str,
    expected_target_center_xyz_m: Sequence[float],
    physical_transactions: int,
    physical_budget: int,
) -> TerminalEvidenceValidation:
    """Rehash all evidence and independently derive one terminal outcome."""

    root = Path(run_root)
    errors: list[str] = []
    hashes: dict[str, str] = {}
    request_ids: list[str] = []
    live_launch = manifest.get("live_launch")
    capability_entrypoint_sha256 = (
        live_launch.get("entrypoint_sha256")
        if isinstance(live_launch, Mapping)
        else None
    )
    try:
        auxiliary_producers = capability_bound_auxiliary_producer_allowlist(
            str(capability_entrypoint_sha256 or "")
        )
    except ValueError:
        auxiliary_producers = {}
    expected_target = _vector(expected_target_center_xyz_m, 3, positive=False)
    expected_simulator = {
        "compiled_model_sha256": expected_compiled_model_sha256,
        "scene_realization_sha256": expected_scene_realization_sha256,
        "scene_fingerprint": expected_scene_fingerprint,
        "scene_execution_evidence_fingerprint": expected_scene_execution_evidence_fingerprint,
    }
    scene_expectation_valid = (
        expected_target is not None
        and all(_sha256(v) for v in expected_simulator.values())
    )

    result_hashes = manifest.get("result_artifact_hashes")
    result_hashes = result_hashes if isinstance(result_hashes, Mapping) else {}
    expected_receipt_hash = result_hashes.get("terminal_verifier_evidence_sha256")
    if not _sha256(expected_receipt_hash):
        return TerminalEvidenceValidation(invalid_reasons=("TERMINAL_EVIDENCE_PIN_MISSING",))
    if not scene_expectation_valid:
        return TerminalEvidenceValidation(invalid_reasons=(
            "TERMINAL_EVIDENCE_SCENE_EXPECTATION_INVALID",
        ))
    assert expected_target is not None
    receipt_path = root / TERMINAL_EVIDENCE_FILENAME
    try:
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise OSError
        raw = receipt_path.read_bytes()
        receipt_hash = hashlib.sha256(raw).hexdigest()
        receipt = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return TerminalEvidenceValidation(invalid_reasons=(
            "TERMINAL_EVIDENCE_MISSING_OR_INVALID",
        ))
    hashes[TERMINAL_EVIDENCE_FILENAME] = receipt_hash
    if receipt_hash != expected_receipt_hash:
        errors.append("TERMINAL_EVIDENCE_HASH_MISMATCH")
    if not isinstance(receipt, Mapping):
        return TerminalEvidenceValidation(receipt_hash, hashes, (), (
            "TERMINAL_EVIDENCE_ROOT_INVALID",
        ))
    if set(receipt) != _RECEIPT_FIELDS:
        errors.append("TERMINAL_EVIDENCE_SCHEMA_FIELDS_INVALID")
    if receipt.get("kind") != TERMINAL_EVIDENCE_KIND:
        errors.append("TERMINAL_EVIDENCE_KIND_INVALID")
    if receipt.get("schema_version") != TERMINAL_EVIDENCE_SCHEMA_VERSION:
        errors.append("TERMINAL_EVIDENCE_SCHEMA_VERSION_INVALID")
    producer = receipt.get("producer")
    if not isinstance(producer, Mapping) or set(producer) != _PRODUCER_FIELDS:
        errors.append("TERMINAL_EVIDENCE_PRODUCER_INVALID")
    elif not _nonempty(producer.get("id")) or not _sha256(producer.get("source_sha256")):
        errors.append("TERMINAL_EVIDENCE_PRODUCER_INVALID")

    expected_binding = {
        "run_id": expected_run_id,
        "seed": expected_seed,
        "condition_fingerprint": expected_condition_fingerprint,
        "terminal_fingerprint": expected_terminal_fingerprint,
        "scene_fingerprint": expected_scene_fingerprint,
        "scene_execution_evidence_fingerprint": expected_scene_execution_evidence_fingerprint,
    }
    binding = receipt.get("run_binding")
    if not isinstance(binding, Mapping) or set(binding) != _RUN_BINDING_FIELDS:
        errors.append("TERMINAL_EVIDENCE_RUN_BINDING_INVALID")
    elif dict(binding) != expected_binding:
        errors.append("TERMINAL_EVIDENCE_RUN_BINDING_MISMATCH")
    status = receipt.get("terminal_status")
    if status not in _VALID_TERMINALS or status != expected_terminal_status:
        errors.append("TERMINAL_EVIDENCE_STATUS_MISMATCH")
    if status in {"SYSTEM_STALL", "CONSECUTIVE_PAYLOAD_DROP_LIMIT"} and not auxiliary_producers:
        errors.append("TERMINAL_AUXILIARY_CAPABILITY_ENTRYPOINT_MISSING")
    if receipt.get("physical_transactions") != physical_transactions:
        errors.append("TERMINAL_EVIDENCE_PHYSICAL_COUNT_MISMATCH")

    state = _load(root, receipt.get("terminal_state"), "terminal_state", errors)
    trace = _load(root, receipt.get("terminal_sample_trace"), "terminal_sample_trace", errors)
    for artifact in (state, trace):
        if artifact is not None:
            hashes[artifact.relative_path] = artifact.sha256
    derived = None
    if trace is not None:
        derived = _derive_trace(
            trace.payload, run_id=expected_run_id, seed=expected_seed,
            expected_simulator=expected_simulator, expected_target=expected_target,
            errors=errors,
        )
    if state is not None:
        _validate_state(
            state.payload, run_id=expected_run_id, seed=expected_seed,
            terminal_status=str(status), expected_simulator=expected_simulator,
            derived=derived, errors=errors,
        )
    checkpoint = derived.checkpoint_id if derived else (_checkpoint(state.payload) if state else "")
    checkpoint_event = (
        derived.checkpoint_event_id
        if derived
        else (_checkpoint_event(state.payload) if state else "")
    )

    stall_ref, drop_ref = receipt.get("system_stall"), receipt.get("payload_drop")
    if status == "SYSTEM_STALL":
        stall = _load(root, stall_ref, "system_stall", errors)
        if stall:
            hashes[stall.relative_path] = stall.sha256
            request_ids.extend(_validate_stall(
                root,
                stall.payload,
                expected_run_id,
                checkpoint,
                checkpoint_event,
                state,
                trace,
                derived,
                auxiliary_producers,
                hashes,
                errors,
            ))
    elif stall_ref is not None:
        errors.append("TERMINAL_EVIDENCE_UNEXPECTED_STALL_LEDGER")
    if status == "CONSECUTIVE_PAYLOAD_DROP_LIMIT":
        drops = _load(root, drop_ref, "payload_drop", errors)
        if drops:
            hashes[drops.relative_path] = drops.sha256
            _validate_drop_ledger(
                root,
                drops.payload,
                expected_run_id,
                checkpoint,
                checkpoint_event,
                physical_transactions,
                expected_compiled_model_sha256,
                expected_scene_realization_sha256,
                auxiliary_producers,
                hashes,
                errors,
            )
    elif drop_ref is not None:
        errors.append("TERMINAL_EVIDENCE_UNEXPECTED_DROP_LEDGER")
    if status == "PHYSICAL_BUDGET_EXHAUSTED" and physical_transactions != physical_budget:
        errors.append("TERMINAL_EVIDENCE_BUDGET_NOT_EXHAUSTED")
    return TerminalEvidenceValidation(
        receipt_hash, hashes, tuple(request_ids), tuple(dict.fromkeys(errors)),
    )


def _load(root: Path, ref: object, label: str, errors: list[str]) -> BoundJsonArtifact | None:
    if ref is None:
        errors.append(f"TERMINAL_EVIDENCE_{label.upper()}_MISSING")
        return None
    try:
        return load_bound_json_artifact(root, ref, label=label)
    except BoundArtifactError:
        errors.append(f"TERMINAL_EVIDENCE_{label.upper()}_INVALID")
        return None


def _validate_simulator(
    value: object, expected: Mapping[str, str], label: str, errors: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != _SIMULATOR_FIELDS:
        errors.append(f"{label}_SIMULATOR_INVALID")
        return None
    if str(value.get("name", "")).strip().lower() != "mujoco":
        errors.append(f"{label}_NOT_MUJOCO")
    if not _nonempty(value.get("version")):
        errors.append(f"{label}_MUJOCO_VERSION_MISSING")
    for key, wanted in expected.items():
        if not _sha256(value.get(key)):
            errors.append(f"{label}_{key.upper()}_INVALID")
        elif value.get(key) != wanted:
            errors.append(f"{label}_{key.upper()}_MISMATCH")
    return value


def _derive_trace(
    value: Mapping[str, Any], *, run_id: str, seed: int,
    expected_simulator: Mapping[str, str], expected_target: tuple[float, ...],
    errors: list[str],
) -> _DerivedTrace | None:
    if set(value) != _TRACE_FIELDS:
        errors.append("TERMINAL_TRACE_SCHEMA_FIELDS_INVALID")
    if (value.get("kind") != MUJOCO_TERMINAL_SAMPLE_TRACE_KIND or
            value.get("schema_version") != MUJOCO_TERMINAL_SAMPLE_TRACE_SCHEMA_VERSION):
        errors.append("TERMINAL_TRACE_IDENTITY_INVALID")
    if value.get("run_id") != run_id or value.get("seed") != seed:
        errors.append("TERMINAL_TRACE_RUN_BINDING_MISMATCH")
    checkpoint = value.get("checkpoint_id")
    if not _nonempty(checkpoint):
        errors.append("TERMINAL_TRACE_CHECKPOINT_MISSING")
        checkpoint = ""
    checkpoint_event = value.get("checkpoint_event_id")
    if not _nonempty(checkpoint_event):
        errors.append("TERMINAL_TRACE_CHECKPOINT_EVENT_MISSING")
        checkpoint_event = ""
    simulator = _validate_simulator(
        value.get("simulator"), expected_simulator, "TERMINAL_TRACE", errors,
    )
    target = value.get("target")
    target_xyz = target_scale = None
    if not isinstance(target, Mapping) or set(target) != _TRACE_TARGET_FIELDS:
        errors.append("TERMINAL_TRACE_TARGET_INVALID")
    else:
        target_xyz = _vector(target.get("center_xyz_m"), 3, positive=False)
        target_scale = _vector(target.get("scale_xyz_m"), 3, positive=True)
        if target_xyz is None:
            errors.append("TERMINAL_TRACE_TARGET_CENTER_INVALID")
        elif not _vectors_close(target_xyz, expected_target):
            errors.append("TERMINAL_TRACE_TARGET_CENTER_MISMATCH")
        if target_scale is None:
            errors.append("TERMINAL_TRACE_TARGET_SCALE_INVALID")
        elif not _vectors_close(target_scale, TARGET_SCALE_XYZ_M):
            errors.append("TERMINAL_TRACE_TARGET_SCALE_MISMATCH")
    geometry = value.get("payload_geometry")
    half_height = None
    if not isinstance(geometry, Mapping) or set(geometry) != _TRACE_PAYLOAD_GEOMETRY_FIELDS:
        errors.append("TERMINAL_TRACE_PAYLOAD_GEOMETRY_INVALID")
    else:
        half_height = _number(geometry.get("half_height_m"))
        if half_height is None or not math.isclose(
            half_height, PAYLOAD_HALF_HEIGHT_M, abs_tol=1e-12, rel_tol=0.0,
        ):
            errors.append("TERMINAL_TRACE_PAYLOAD_HALF_HEIGHT_MISMATCH")
    lifecycle = value.get("lifecycle")
    if (not isinstance(lifecycle, Mapping) or set(lifecycle) != _LIFECYCLE_FIELDS or
            any(not isinstance(lifecycle.get(k), bool) for k in _LIFECYCLE_FIELDS)):
        errors.append("TERMINAL_TRACE_LIFECYCLE_INVALID")
        lifecycle_value = {key: False for key in _LIFECYCLE_FIELDS}
    else:
        lifecycle_value = {key: bool(lifecycle[key]) for key in _LIFECYCLE_FIELDS}
    declared_window = _number(value.get("sample_window_s"))
    if declared_window is None or declared_window < 0:
        errors.append("TERMINAL_TRACE_SAMPLE_WINDOW_INVALID")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("TERMINAL_TRACE_SAMPLES_INVALID")
        return None
    if simulator is None or target_xyz is None or target_scale is None or half_height is None:
        return None

    released: list[bool] = []
    contact_counts = {role: 0 for role in ("target", "floor", "wall", "robot")}
    xy, z, payload_tilt, times = [], [], [], []
    heights = {rid: [] for rid in ("front", "back")}
    distances = {rid: [] for rid in ("front", "back")}
    robot_tilt = {rid: [] for rid in ("front", "back")}
    target_payload_z = target_xyz[2] + abs(target_scale[2]) / 2 + half_height
    valid_samples = 0
    for index, sample in enumerate(samples):
        prefix = f"TERMINAL_TRACE_SAMPLE_{index}"
        if not isinstance(sample, Mapping) or set(sample) != _TRACE_SAMPLE_FIELDS:
            errors.append(f"{prefix}_SCHEMA_INVALID")
            continue
        if sample.get("sample_index") != index:
            errors.append(f"{prefix}_INDEX_INVALID")
        time_s = _number(sample.get("time_s"))
        if time_s is None or (times and time_s <= times[-1]):
            errors.append(f"{prefix}_TIME_INVALID")
        else:
            times.append(time_s)
        sample_released = sample.get("payload_released")
        if not isinstance(sample_released, bool):
            errors.append(f"{prefix}_RELEASED_INVALID")
            sample_released = False
        payload_pose = _pose(sample.get("payload_pose"), f"{prefix}_PAYLOAD_POSE", errors)
        robot_mapping = sample.get("robot_poses")
        poses = {}
        if not isinstance(robot_mapping, Mapping) or set(robot_mapping) != {"front", "back"}:
            errors.append(f"{prefix}_ROBOT_POSES_INVALID")
        else:
            for rid in ("front", "back"):
                pose = _pose(robot_mapping.get(rid), f"{prefix}_ROBOT_{rid.upper()}_POSE", errors)
                if pose:
                    poses[rid] = pose
        roles = _contact_roles(sample.get("contacts"), prefix, errors)
        if payload_pose is None or len(poses) != 2 or time_s is None:
            continue
        valid_samples += 1
        position, tilt = payload_pose
        released.append(sample_released)
        contact_counts["target"] += "target" in roles
        contact_counts["floor"] += "floor" in roles
        contact_counts["wall"] += "wall" in roles
        contact_counts["robot"] += bool({"front_robot", "back_robot"} & roles)
        xy.append(math.hypot(position[0] - target_xyz[0], position[1] - target_xyz[1]))
        z.append(abs(position[2] - target_payload_z))
        payload_tilt.append(tilt)
        for rid, (robot_position, tilt_value) in poses.items():
            heights[rid].append(robot_position[2])
            distances[rid].append(math.hypot(
                robot_position[0] - target_xyz[0], robot_position[1] - target_xyz[1],
            ))
            robot_tilt[rid].append(tilt_value)
    if valid_samples != len(samples) or len(times) != len(samples):
        errors.append("TERMINAL_TRACE_INCOMPLETE_VALID_SAMPLE_SET")
        return None
    actual_window = times[-1] - times[0] if len(times) > 1 else 0.0
    if declared_window is None or not math.isclose(
        declared_window, actual_window, rel_tol=0.0, abs_tol=1e-6,
    ):
        errors.append("TERMINAL_TRACE_SAMPLE_WINDOW_MISMATCH")
    robots = {
        rid: _DerivedRobot(
            upright=max(robot_tilt[rid]) <= MAX_UPRIGHT_ROBOT_TILT_DEG,
            pelvis_height_m=min(heights[rid]),
            distance_to_target_m=min(distances[rid]),
        ) for rid in ("front", "back")
    }
    return _DerivedTrace(
        checkpoint_id=str(checkpoint),
        checkpoint_event_id=str(checkpoint_event),
        simulator=dict(simulator),
        sample_window_s=actual_window, sample_count=len(samples),
        payload=_DerivedPayload(
            released=all(released),
            target_contact_all_samples=contact_counts["target"] == len(samples),
            floor_contact_samples=contact_counts["floor"],
            wall_contact_samples=contact_counts["wall"],
            robot_contact_samples=contact_counts["robot"],
            xy_error_m=max(xy), z_error_m=max(z), tilt_deg=max(payload_tilt),
        ),
        robots=robots, lifecycle=lifecycle_value,
    )


def _validate_state(
    value: Mapping[str, Any], *, run_id: str, seed: int, terminal_status: str,
    expected_simulator: Mapping[str, str], derived: _DerivedTrace | None,
    errors: list[str],
) -> None:
    """Validate the runner summary only after deriving the raw-trace values."""

    if set(value) != _STATE_FIELDS:
        errors.append("TERMINAL_STATE_SCHEMA_FIELDS_INVALID")
    if (value.get("kind") != MUJOCO_TERMINAL_STATE_KIND or
            value.get("schema_version") != MUJOCO_TERMINAL_STATE_SCHEMA_VERSION):
        errors.append("TERMINAL_STATE_IDENTITY_INVALID")
    if value.get("run_id") != run_id or value.get("seed") != seed:
        errors.append("TERMINAL_STATE_RUN_BINDING_MISMATCH")
    checkpoint = value.get("checkpoint_id")
    if not _nonempty(checkpoint):
        errors.append("TERMINAL_STATE_CHECKPOINT_MISSING")
    checkpoint_event = value.get("checkpoint_event_id")
    if not _nonempty(checkpoint_event):
        errors.append("TERMINAL_STATE_CHECKPOINT_EVENT_MISSING")
    _validate_simulator(value.get("simulator"), expected_simulator, "TERMINAL_STATE", errors)
    if derived is None:
        errors.append("TERMINAL_STATE_RAW_TRACE_UNAVAILABLE")
        return
    if checkpoint != derived.checkpoint_id:
        errors.append("TERMINAL_STATE_TRACE_CHECKPOINT_MISMATCH")
    if checkpoint_event != derived.checkpoint_event_id:
        errors.append("TERMINAL_STATE_TRACE_CHECKPOINT_EVENT_MISMATCH")
    if value.get("simulator") != derived.simulator:
        errors.append("TERMINAL_STATE_TRACE_SIMULATOR_MISMATCH")
    if not _close(value.get("sample_window_s"), derived.sample_window_s):
        errors.append("TERMINAL_STATE_TRACE_SAMPLE_WINDOW_MISMATCH")
    if value.get("sample_count") != derived.sample_count:
        errors.append("TERMINAL_STATE_TRACE_SAMPLE_COUNT_MISMATCH")

    payload = value.get("payload")
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_FIELDS:
        errors.append("TERMINAL_STATE_PAYLOAD_MEASUREMENTS_INVALID")
    else:
        expected_payload: Mapping[str, Any] = {
            "released": derived.payload.released,
            "target_contact_all_samples": derived.payload.target_contact_all_samples,
            "floor_contact_samples": derived.payload.floor_contact_samples,
            "wall_contact_samples": derived.payload.wall_contact_samples,
            "robot_contact_samples": derived.payload.robot_contact_samples,
            "xy_error_m": derived.payload.xy_error_m,
            "z_error_m": derived.payload.z_error_m,
            "tilt_deg": derived.payload.tilt_deg,
        }
        for key, wanted in expected_payload.items():
            actual = payload.get(key)
            matches = (_close(actual, wanted) if isinstance(wanted, float)
                       else type(actual) is type(wanted) and actual == wanted)
            if not matches:
                errors.append(f"TERMINAL_STATE_PAYLOAD_{key.upper()}_TRACE_MISMATCH")

    robots = value.get("robots")
    if not isinstance(robots, Mapping) or set(robots) != {"front", "back"}:
        errors.append("TERMINAL_STATE_ROBOT_MEASUREMENTS_INVALID")
    else:
        for rid in ("front", "back"):
            robot = robots.get(rid)
            prefix = f"TERMINAL_STATE_ROBOT_{rid.upper()}"
            if not isinstance(robot, Mapping) or set(robot) != _ROBOT_FIELDS:
                errors.append(f"{prefix}_MEASUREMENTS_INVALID")
                continue
            wanted = derived.robots[rid]
            comparisons = {
                "upright": type(robot.get("upright")) is bool and robot.get("upright") == wanted.upright,
                "pelvis_height_m": _close(robot.get("pelvis_height_m"), wanted.pelvis_height_m),
                "distance_to_target_m": _close(
                    robot.get("distance_to_target_m"), wanted.distance_to_target_m,
                ),
            }
            errors.extend(
                f"{prefix}_{key.upper()}_TRACE_MISMATCH"
                for key, matches in comparisons.items() if not matches
            )
    lifecycle = value.get("lifecycle")
    if (not isinstance(lifecycle, Mapping) or set(lifecycle) != _LIFECYCLE_FIELDS or
            any(not isinstance(lifecycle.get(k), bool) for k in _LIFECYCLE_FIELDS)):
        errors.append("TERMINAL_STATE_LIFECYCLE_INVALID")
    elif dict(lifecycle) != dict(derived.lifecycle):
        errors.append("TERMINAL_STATE_LIFECYCLE_TRACE_MISMATCH")

    if terminal_status == "SYSTEM_STALL":
        if derived.payload.floor_contact_samples or derived.payload.wall_contact_samples:
            errors.append("SYSTEM_STALL_TERMINAL_CHECKPOINT_UNSAFE")
        for rid, robot in derived.robots.items():
            if not robot.upright or robot.pelvis_height_m < TRUSTED_CHECKPOINT_MIN_PELVIS_HEIGHT_M:
                errors.append(f"SYSTEM_STALL_ROBOT_{rid.upper()}_CHECKPOINT_INVALID")
    if terminal_status != "SUCCESS":
        return
    if derived.sample_window_s + 1e-9 < TERMINAL_STABLE_WINDOW_S:
        errors.append("SUCCESS_STABILITY_WINDOW_TOO_SHORT")
    if derived.sample_count < 2:
        errors.append("SUCCESS_STABILITY_SAMPLE_COUNT_INVALID")
    checks = (
        (derived.payload.released, "SUCCESS_PAYLOAD_NOT_RELEASED"),
        (derived.payload.target_contact_all_samples, "SUCCESS_TARGET_SUPPORT_NOT_STABLE"),
        (derived.payload.floor_contact_samples == 0, "SUCCESS_PAYLOAD_FLOOR_CONTACT"),
        (derived.payload.wall_contact_samples == 0, "SUCCESS_PAYLOAD_WALL_CONTACT"),
        (derived.payload.robot_contact_samples == 0, "SUCCESS_PAYLOAD_STILL_ROBOT_SUPPORTED"),
        (derived.payload.xy_error_m <= TARGET_SUPPORT_RADIUS_M,
         "SUCCESS_PAYLOAD_OUTSIDE_TARGET_SUPPORT"),
        (derived.payload.z_error_m <= TARGET_Z_TOLERANCE_M, "SUCCESS_PAYLOAD_Z_INVALID"),
        (derived.payload.tilt_deg <= MAX_TERMINAL_PAYLOAD_TILT_DEG,
         "SUCCESS_PAYLOAD_TILT_INVALID"),
    )
    errors.extend(reason for passed, reason in checks if not passed)
    for rid, robot in derived.robots.items():
        prefix = f"SUCCESS_ROBOT_{rid.upper()}"
        if not robot.upright:
            errors.append(f"{prefix}_NOT_UPRIGHT")
        if robot.pelvis_height_m < UPRIGHT_MIN_PELVIS_HEIGHT_M:
            errors.append(f"{prefix}_PELVIS_TOO_LOW")
        if robot.distance_to_target_m < ROBOT_EXIT_MIN_DISTANCE_M:
            errors.append(f"{prefix}_TARGET_CLEARANCE_INVALID")
    for key in _LIFECYCLE_FIELDS:
        if derived.lifecycle.get(key) is not True:
            errors.append(f"SUCCESS_{key.upper()}_MISSING")


def _validate_stall(
    root: Path,
    value: Mapping[str, Any],
    run_id: str,
    terminal_checkpoint: str,
    terminal_checkpoint_event: str,
    state: BoundJsonArtifact | None,
    trace: BoundJsonArtifact | None,
    derived: _DerivedTrace | None,
    expected_producers: Mapping[str, Mapping[str, str]],
    hashes: dict[str, str],
    errors: list[str],
) -> tuple[str, ...]:
    if state is None or trace is None or derived is None:
        errors.append("SYSTEM_STALL_RAW_TERMINAL_CHECKPOINT_UNAVAILABLE")
        return ()
    checkpoint_summary = {
        "sample_window_s": derived.sample_window_s,
        "sample_count": derived.sample_count,
        "payload": {
            "floor_contact_samples": derived.payload.floor_contact_samples,
            "wall_contact_samples": derived.payload.wall_contact_samples,
        },
        "robots": {
            robot_id: {
                "upright": robot.upright,
                "pelvis_height_m": robot.pelvis_height_m,
            }
            for robot_id, robot in derived.robots.items()
        },
    }
    result = recompute_system_stall_ledger(
        root,
        source_index=value,
        expected_run_id=run_id,
        expected_checkpoint_id=terminal_checkpoint,
        expected_checkpoint_event_id=terminal_checkpoint_event,
        expected_terminal_state_sha256=state.sha256,
        expected_terminal_sample_trace_sha256=trace.sha256,
        derived_checkpoint=checkpoint_summary,
        expected_producers=expected_producers,
    )
    hashes.update(result.artifact_hashes or {})
    errors.extend(result.invalid_reasons)
    if result.ledger is None or dict(value) != dict(result.ledger):
        errors.append("SYSTEM_STALL_LEDGER_NOT_RECOMPUTED_FROM_RAW_ARTIFACTS")
    return result.request_ids


def _validate_drop_ledger(
    root: Path,
    value: Mapping[str, Any],
    run_id: str,
    terminal_checkpoint: str,
    terminal_checkpoint_event: str,
    physical_transactions: int,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    expected_producers: Mapping[str, Mapping[str, str]],
    hashes: dict[str, str],
    errors: list[str],
) -> None:
    result = recompute_payload_drop_ledger(
        root,
        source_index=value,
        expected_run_id=run_id,
        expected_terminal_checkpoint_id=terminal_checkpoint,
        expected_terminal_checkpoint_event_id=terminal_checkpoint_event,
        expected_compiled_model_sha256=expected_compiled_model_sha256,
        expected_scene_realization_sha256=expected_scene_realization_sha256,
        expected_producers=expected_producers,
        physical_transactions=physical_transactions,
    )
    hashes.update(result.artifact_hashes or {})
    errors.extend(result.invalid_reasons)
    if result.ledger is None or dict(value) != dict(result.ledger):
        errors.append("PAYLOAD_DROP_LEDGER_NOT_RECOMPUTED_FROM_RAW_RECEIPTS")


def _contact_roles(value: object, prefix: str, errors: list[str]) -> set[str]:
    if not isinstance(value, list):
        errors.append(f"{prefix}_CONTACTS_INVALID")
        return set()
    result = set()
    for index, contact in enumerate(value):
        label = f"{prefix}_CONTACT_{index}"
        if not isinstance(contact, Mapping) or set(contact) != _CONTACT_FIELDS:
            errors.append(f"{label}_SCHEMA_INVALID")
            continue
        geom1, geom2 = _integer(contact.get("geom1_id")), _integer(contact.get("geom2_id"))
        role1, role2, distance = (
            contact.get("geom1_role"), contact.get("geom2_role"), _number(contact.get("distance_m")),
        )
        if geom1 is None or geom2 is None or min(geom1, geom2) < 0 or geom1 == geom2:
            errors.append(f"{label}_GEOM_ID_INVALID")
        if role1 not in _CONTACT_ROLES or role2 not in _CONTACT_ROLES or role1 == role2:
            errors.append(f"{label}_ROLE_INVALID")
            continue
        if distance is None:
            errors.append(f"{label}_DISTANCE_INVALID")
        elif distance <= CONTACT_DISTANCE_TOLERANCE_M and "payload" in {role1, role2}:
            result.update({str(role1), str(role2)} - {"payload"})
    return result


def _pose(value: object, label: str, errors: list[str]) -> tuple[tuple[float, ...], float] | None:
    if not isinstance(value, Mapping) or set(value) != _POSE_FIELDS:
        errors.append(f"{label}_INVALID")
        return None
    position = _vector(value.get("position_xyz_m"), 3, positive=False)
    quaternion = _vector(value.get("quaternion_wxyz"), 4, positive=False)
    if position is None:
        errors.append(f"{label}_POSITION_INVALID")
    if quaternion is None:
        errors.append(f"{label}_QUATERNION_INVALID")
    if position is None or quaternion is None:
        return None
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        errors.append(f"{label}_QUATERNION_NOT_UNIT")
        return None
    w, x, y, z = (component / norm for component in quaternion)
    del w, z
    up_dot = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return position, math.degrees(math.acos(up_dot))


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _vector(value: object, length: int, *, positive: bool) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result = tuple(_number(item) for item in value)
    if any(item is None for item in result):
        return None
    numbers = tuple(float(item) for item in result if item is not None)
    if positive and any(item <= 0 for item in numbers):
        return None
    return numbers


def _vectors_close(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9) for a, b in zip(left, right)
    )


def _close(value: object, expected: float) -> bool:
    number = _number(value)
    return number is not None and math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-6)


def _checkpoint(value: Mapping[str, Any]) -> str:
    checkpoint = value.get("checkpoint_id")
    return checkpoint.strip() if isinstance(checkpoint, str) else ""


def _checkpoint_event(value: Mapping[str, Any]) -> str:
    checkpoint_event = value.get("checkpoint_event_id")
    return checkpoint_event.strip() if isinstance(checkpoint_event, str) else ""


__all__ = [
    "CONTACT_DISTANCE_TOLERANCE_M", "MAX_TERMINAL_PAYLOAD_TILT_DEG",
    "MUJOCO_TERMINAL_SAMPLE_TRACE_KIND", "MUJOCO_TERMINAL_SAMPLE_TRACE_SCHEMA_VERSION",
    "MUJOCO_TERMINAL_STATE_KIND", "MUJOCO_TERMINAL_STATE_SCHEMA_VERSION",
    "PAYLOAD_DROP_LEDGER_KIND", "PAYLOAD_DROP_LEDGER_SCHEMA_VERSION", "PAYLOAD_DROP_LIMIT",
    "ROBOT_EXIT_MIN_DISTANCE_M", "SYSTEM_STALL_LEDGER_KIND",
    "SYSTEM_STALL_LEDGER_SCHEMA_VERSION", "SYSTEM_STALL_GENERATION_CYCLES",
    "TARGET_SUPPORT_RADIUS_M", "TARGET_Z_TOLERANCE_M", "TERMINAL_EVIDENCE_FILENAME",
    "TERMINAL_EVIDENCE_KIND", "TERMINAL_EVIDENCE_SCHEMA_VERSION",
    "TERMINAL_STABLE_WINDOW_S", "TerminalEvidenceValidation",
    "UPRIGHT_MIN_PELVIS_HEIGHT_M", "validate_terminal_evidence",
]
