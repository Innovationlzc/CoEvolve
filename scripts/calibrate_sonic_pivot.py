#!/usr/bin/env python3
"""Calibrate the loaded coordinated-pivot primitive without model calls."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import threading
import time
from math import degrees, hypot, pi
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.cap_policy.backends import StaticSonicSkillBackend
from praxis_coop.cap_policy.checkpoints import SimulatorCheckpointClient
from praxis_coop.cap_policy.contracts import write_json
from praxis_coop.cap_policy.scene_clearance import (
    authored_scene_motion_clearance,
    pivot_role_assignment,
)
from praxis_coop.sonic.runtime_state import (
    read_payload_runtime_status,
)
from praxis_coop.sonic_mujoco.manual_layout_scene import (
    DEFAULT_USDA_SCENE,
)

DEFAULT_ANGLES_DEG = (10.0, 15.0, 20.0, 30.0)
DEFAULT_TURN_DIRECTIONS = ("clockwise", "counterclockwise")
ROLE_ASSIGNMENTS = (
    {"front": "inner", "back": "outer"},
    {"front": "outer", "back": "inner"},
)
CSV_FIELDS = (
    "sample_id",
    "turn_direction",
    "requested_angle_deg",
    "front_role",
    "back_role",
    "geometry_role_match",
    "repetition",
    "backend_status",
    "actual_payload_yaw_deg",
    "signed_directional_yaw_deg",
    "yaw_gain",
    "semantic_fidelity",
    "front_displacement_m",
    "back_displacement_m",
    "min_front_support_force_n",
    "min_back_support_force_n",
    "min_support_force_balance_ratio",
    "bilateral_support_sample_ratio",
    "min_robot_pelvis_height_m",
    "max_payload_tilt_deg",
    "wall_contact_observed",
    "floor_contact_observed",
    "min_commanded_turn_clearance_m",
    "stable",
    "failure_reason",
)


class RuntimeSampler:
    """Poll authoritative Term1 state while one blocking backend call runs."""

    def __init__(self, runtime_state_file: Path, *, period_s: float = 0.05) -> None:
        self.runtime_state_file = Path(runtime_state_file)
        self.period_s = max(0.02, float(period_s))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            state = _read_json(self.runtime_state_file)
            if state:
                self.samples.append(_sample_runtime_state(state))
            self._stop.wait(self.period_s)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "calibration_config.json", _config_payload(args))

    backend = StaticSonicSkillBackend(
        execute=True,
        front_port=args.front_port,
        back_port=args.back_port,
        default_start_pulse_seconds=0.0,
        default_stand_seconds=args.pre_stand_s,
        post_transport_idle_s=args.post_idle_s,
        payload_safety_profile=args.payload_safety_profile,
        stream_output=not args.quiet,
    )
    checkpoint_client = SimulatorCheckpointClient(
        heartbeat_file=args.heartbeat_file,
        runtime_state_file=args.runtime_state_file,
        artifact_dir=args.output_dir / "checkpoint_events",
        timeout_s=args.checkpoint_timeout_s,
        restore_settle_s=args.checkpoint_settle_s,
    )

    _prepare_loaded_baseline(args, backend, checkpoint_client)
    samples: list[dict[str, Any]] = []
    call_index = 0
    for direction in args.turn_directions:
        for roles in ROLE_ASSIGNMENTS:
            for angle_deg in args.angles_deg:
                for repetition in range(1, args.repetitions + 1):
                    call_index += 1
                    sample = _run_sample(
                        args=args,
                        backend=backend,
                        checkpoint_client=checkpoint_client,
                        call_index=call_index,
                        direction=direction,
                        roles=roles,
                        angle_deg=angle_deg,
                        repetition=repetition,
                    )
                    samples.append(sample)
                    write_json(
                        args.output_dir / "pivot_calibration_samples.json", samples
                    )
                    _write_csv(
                        args.output_dir / "pivot_calibration_samples.csv", samples
                    )
                    print(
                        "[pivot-calibration] "
                        f"{sample['sample_id']} stable={sample['stable']} "
                        f"requested={angle_deg:g}deg "
                        f"actual={sample['actual_payload_yaw_deg']:.2f}deg "
                        f"support=({sample['min_front_support_force_n']:.2f},"
                        f"{sample['min_back_support_force_n']:.2f})N",
                        flush=True,
                    )

    summary = _summarize(args, samples)
    write_json(args.output_dir / "pivot_calibration_summary.json", summary)
    print(
        "[pivot-calibration] complete "
        f"stable={summary['stable_samples']}/{summary['sample_count']} "
        f"allowlist={summary['recommended_model_angle_allowlist_deg']}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heartbeat-file", type=Path, required=True)
    parser.add_argument("--runtime-state-file", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint-id",
        default="accepted_step_00",
        help="Stable robot checkpoint used before guarded payload placement.",
    )
    parser.add_argument(
        "--baseline-checkpoint-id",
        default="pivot_calibration_loaded",
        help="Loaded checkpoint restored before every calibration sample.",
    )
    parser.add_argument(
        "--reuse-loaded-baseline",
        action="store_true",
        help="Skip source restore and placement when the loaded baseline already exists.",
    )
    parser.add_argument(
        "--angles-deg", type=float, nargs="+", default=DEFAULT_ANGLES_DEG
    )
    parser.add_argument(
        "--turn-directions",
        nargs="+",
        choices=DEFAULT_TURN_DIRECTIONS,
        default=DEFAULT_TURN_DIRECTIONS,
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--swing-radius-m", type=float, default=0.8)
    parser.add_argument("--front-port", type=int, default=5556)
    parser.add_argument("--back-port", type=int, default=5558)
    parser.add_argument("--pre-stand-s", type=float, default=0.5)
    parser.add_argument("--post-idle-s", type=float, default=1.0)
    parser.add_argument("--planner-resync-s", type=float, default=2.5)
    parser.add_argument("--checkpoint-timeout-s", type=float, default=12.0)
    parser.add_argument("--checkpoint-settle-s", type=float, default=4.0)
    parser.add_argument("--loaded-state-timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--payload-safety-profile", choices=("task", "strict"), default="task"
    )
    parser.add_argument("--scene-usda", type=Path, default=DEFAULT_USDA_SCENE)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    args.angles_deg = tuple(sorted({float(value) for value in args.angles_deg}))
    if not args.angles_deg or any(value <= 0.0 for value in args.angles_deg):
        parser.error("--angles-deg values must be positive")
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.duration_s <= 0.0 or args.swing_radius_m <= 0.0:
        parser.error("--duration-s and --swing-radius-m must be positive")
    return args


def _prepare_loaded_baseline(
    args: argparse.Namespace,
    backend: StaticSonicSkillBackend,
    checkpoint_client: SimulatorCheckpointClient,
) -> None:
    state = _read_json(args.runtime_state_file)
    available = _available_checkpoint_ids(state)
    if args.reuse_loaded_baseline:
        if args.baseline_checkpoint_id not in available:
            raise RuntimeError(
                f"loaded baseline {args.baseline_checkpoint_id!r} is unavailable: {available}"
            )
        _restore_exact(
            args,
            backend,
            checkpoint_client,
            args.baseline_checkpoint_id,
            "baseline_probe",
        )
        loaded = _wait_for_loaded_state(
            args.runtime_state_file,
            timeout_s=args.loaded_state_timeout_s,
        )
        write_json(args.output_dir / "loaded_baseline_state.json", loaded)
        return

    if args.source_checkpoint_id not in available:
        raise RuntimeError(
            f"source checkpoint {args.source_checkpoint_id!r} is unavailable: {available}"
        )
    _restore_exact(
        args, backend, checkpoint_client, args.source_checkpoint_id, "source"
    )
    placement = backend.execute_skill(
        skill_name="place_payload_on_arms",
        parameters={
            "place_hold_seconds": 2.0,
            "stand_seconds": 0.5,
            "final_idle_seconds": 0.5,
            "payload_force": True,
        },
        output_dir=args.output_dir / "baseline_initialization",
        heartbeat_file=args.heartbeat_file,
        runtime_state_file=args.runtime_state_file,
        call_index=0,
    )
    if placement.status != "success":
        raise RuntimeError(f"loaded baseline placement failed: {placement.status}")
    loaded = _wait_for_loaded_state(
        args.runtime_state_file,
        timeout_s=args.loaded_state_timeout_s,
    )
    checkpoint_client.save(args.baseline_checkpoint_id, reset_metrics=True)
    write_json(args.output_dir / "loaded_baseline_state.json", loaded)


def _restore_exact(
    args: argparse.Namespace,
    backend: StaticSonicSkillBackend,
    checkpoint_client: SimulatorCheckpointClient,
    checkpoint_id: str,
    label: str,
) -> dict[str, Any]:
    checkpoint_client.restore(checkpoint_id)
    backend.resynchronize_idle(
        output_dir=args.output_dir / "planner_resync" / label,
        heartbeat_file=args.heartbeat_file,
        runtime_state_file=args.runtime_state_file,
        duration_s=args.planner_resync_s,
        stand_phase_id=f"pivot_calibration_{label}_resync",
    )
    restored = checkpoint_client.restore(checkpoint_id)
    _wait_for_checkpoint_release(
        args.runtime_state_file,
        checkpoint_id=checkpoint_id,
        timeout_s=max(args.checkpoint_timeout_s, args.checkpoint_settle_s + 4.0),
    )
    return restored.runtime_state_after or _read_json(args.runtime_state_file)


def _run_sample(
    *,
    args: argparse.Namespace,
    backend: StaticSonicSkillBackend,
    checkpoint_client: SimulatorCheckpointClient,
    call_index: int,
    direction: str,
    roles: dict[str, str],
    angle_deg: float,
    repetition: int,
) -> dict[str, Any]:
    sample_id = (
        f"{direction}_{roles['front']}-{roles['back']}_"
        f"{angle_deg:04.1f}deg_r{repetition:02d}"
    ).replace(".", "p")
    sample_dir = args.output_dir / "samples" / sample_id
    _restore_exact(
        args,
        backend,
        checkpoint_client,
        args.baseline_checkpoint_id,
        sample_id,
    )
    before = _wait_for_loaded_state(
        args.runtime_state_file,
        timeout_s=args.loaded_state_timeout_s,
    )
    before_clearance = authored_scene_motion_clearance(
        before,
        source_usda=args.scene_usda,
    )
    geometry_roles = pivot_role_assignment(before, direction)
    geometry_role_match = bool(
        geometry_roles
        and all(
            roles.get(robot_id) == role for robot_id, role in geometry_roles.items()
        )
    )
    parameters = {
        robot_id: {
            "turn_direction": direction,
            "travel_direction": "forward",
            "angle_deg": angle_deg,
            "duration_s": args.duration_s,
            "role": roles[robot_id],
            "swing_radius_m": args.swing_radius_m,
        }
        for robot_id in ("front", "back")
    }
    actions = {
        robot_id: {
            "method": "coordinated_pivot",
            "parameters": values,
        }
        for robot_id, values in parameters.items()
    }
    sampler = RuntimeSampler(args.runtime_state_file)
    sampler.start()
    try:
        record = backend.execute_team_actions(
            actions_by_robot=actions,
            phase_id=f"pivot_calibration_{call_index:03d}",
            synchronization_group=f"pivot_calibration_{call_index:03d}",
            coordination_mode="async",
            output_dir=sample_dir,
            heartbeat_file=args.heartbeat_file,
            runtime_state_file=args.runtime_state_file,
            call_index=call_index,
        )
    finally:
        runtime_samples = sampler.stop()
    after = _read_json(args.runtime_state_file)
    after_clearance = authored_scene_motion_clearance(
        after,
        source_usda=args.scene_usda,
    )
    result = _read_json(sample_dir / "coordinated_team_result.json")
    sample = _build_sample_result(
        sample_id=sample_id,
        direction=direction,
        roles=roles,
        geometry_role_match=geometry_role_match,
        angle_deg=angle_deg,
        repetition=repetition,
        record_status=record.status,
        before=before,
        after=after,
        runtime_samples=runtime_samples,
        coordinated_result=result,
        before_clearance=before_clearance,
        after_clearance=after_clearance,
    )
    write_json(sample_dir / "calibration_measurement.json", sample)
    return sample


def _build_sample_result(
    *,
    sample_id: str,
    direction: str,
    roles: dict[str, str],
    geometry_role_match: bool,
    angle_deg: float,
    repetition: int,
    record_status: str,
    before: dict[str, Any],
    after: dict[str, Any],
    runtime_samples: list[dict[str, Any]],
    coordinated_result: dict[str, Any],
    before_clearance: dict[str, Any],
    after_clearance: dict[str, Any],
) -> dict[str, Any]:
    before_yaw = _payload_yaw(before)
    after_yaw = _payload_yaw(after)
    actual_yaw_deg = degrees(_wrapped_angle(after_yaw - before_yaw))
    direction_sign = -1.0 if direction == "clockwise" else 1.0
    signed_yaw_deg = direction_sign * actual_yaw_deg
    yaw_gain = signed_yaw_deg / angle_deg
    semantic_fidelity = 0.5 <= yaw_gain <= 1.5
    sampled = runtime_samples or [_sample_runtime_state(after)]
    support_count = sum(bool(item["bilateral_support"]) for item in sampled)
    min_front_force = min(float(item["front_support_force_n"]) for item in sampled)
    min_back_force = min(float(item["back_support_force_n"]) for item in sampled)
    min_balance = min(float(item["support_force_balance_ratio"]) for item in sampled)
    min_pelvis = min(float(item["min_robot_pelvis_height_m"]) for item in sampled)
    max_tilt = max(float(item["payload_tilt_deg"]) for item in sampled)
    wall_contact = any(bool(item["wall_contact"]) for item in sampled)
    floor_contact = any(bool(item["floor_contact"]) for item in sampled)
    support_ratio = support_count / max(1, len(sampled))
    backend_error = str(coordinated_result.get("error", "")).strip()
    final_support = _supporting_robots(after)
    failure_reasons: list[str] = []
    if record_status != "success" or coordinated_result.get("status") != "success":
        failure_reasons.append(backend_error or record_status)
    if signed_yaw_deg <= 0.0:
        failure_reasons.append("payload yaw moved opposite the requested direction")
    if floor_contact:
        failure_reasons.append("payload floor contact")
    if wall_contact:
        failure_reasons.append("payload wall contact")
    if not {"front", "back"}.issubset(final_support):
        failure_reasons.append("final bilateral support missing")
    if min_pelvis < 0.50:
        failure_reasons.append("robot fall")
    stable = not failure_reasons
    clearance_values = [
        _turn_clearance(before_clearance, direction),
        _turn_clearance(after_clearance, direction),
    ]
    finite_clearances = [value for value in clearance_values if value is not None]
    return {
        "sample_id": sample_id,
        "turn_direction": direction,
        "requested_angle_deg": angle_deg,
        "front_role": roles["front"],
        "back_role": roles["back"],
        "geometry_role_match": geometry_role_match,
        "repetition": repetition,
        "backend_status": record_status,
        "backend_error": backend_error,
        "actual_payload_yaw_deg": actual_yaw_deg,
        "signed_directional_yaw_deg": signed_yaw_deg,
        "yaw_gain": yaw_gain,
        "semantic_fidelity": semantic_fidelity,
        "front_displacement_m": _robot_displacement(before, after, "front"),
        "back_displacement_m": _robot_displacement(before, after, "back"),
        "min_front_support_force_n": min_front_force,
        "min_back_support_force_n": min_back_force,
        "min_support_force_balance_ratio": min_balance,
        "bilateral_support_sample_ratio": support_ratio,
        "min_robot_pelvis_height_m": min_pelvis,
        "max_payload_tilt_deg": max_tilt,
        "wall_contact_observed": wall_contact,
        "floor_contact_observed": floor_contact,
        "min_commanded_turn_clearance_m": (
            min(finite_clearances) if finite_clearances else None
        ),
        "stable": stable,
        "failure_reason": "; ".join(failure_reasons),
        "runtime_sample_count": len(sampled),
        "before_pose": _compact_pose(before),
        "after_pose": _compact_pose(after),
        "before_turn_clearance": _turn_clearance_record(before_clearance, direction),
        "after_turn_clearance": _turn_clearance_record(after_clearance, direction),
    }


def _summarize(
    args: argparse.Namespace, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    allowlist: list[float] = []
    for angle_deg in args.angles_deg:
        angle_samples = [
            item for item in samples if float(item["requested_angle_deg"]) == angle_deg
        ]
        geometry_samples = [
            item for item in angle_samples if bool(item["geometry_role_match"])
        ]
        all_stable = bool(geometry_samples) and all(
            bool(item["stable"]) for item in geometry_samples
        )
        all_semantic = bool(geometry_samples) and all(
            bool(item["semantic_fidelity"]) for item in geometry_samples
        )
        gains = [float(item["yaw_gain"]) for item in angle_samples]
        signed_yaws = [
            float(item["signed_directional_yaw_deg"]) for item in angle_samples
        ]
        groups.append(
            {
                "requested_angle_deg": angle_deg,
                "sample_count": len(angle_samples),
                "stable_count": sum(bool(item["stable"]) for item in angle_samples),
                "geometry_matched_sample_count": len(geometry_samples),
                "all_geometry_matched_samples_stable": all_stable,
                "all_geometry_matched_samples_semantically_faithful": all_semantic,
                "mean_signed_payload_yaw_deg": statistics.fmean(signed_yaws),
                "stdev_signed_payload_yaw_deg": (
                    statistics.stdev(signed_yaws) if len(signed_yaws) > 1 else 0.0
                ),
                "mean_yaw_gain": statistics.fmean(gains),
                "max_yaw_gain": max(gains),
                "min_bilateral_support_sample_ratio": min(
                    float(item["bilateral_support_sample_ratio"])
                    for item in angle_samples
                ),
                "min_support_force_per_robot_n": min(
                    min(
                        float(item["min_front_support_force_n"]),
                        float(item["min_back_support_force_n"]),
                    )
                    for item in angle_samples
                ),
            }
        )
        if all_stable and all_semantic:
            allowlist.append(angle_deg)
    return {
        "model_calls": 0,
        "calibration_scope": (
            "loaded coordinated_pivot restored from one identical physics baseline "
            "before every sample"
        ),
        "baseline_checkpoint_id": args.baseline_checkpoint_id,
        "duration_s": args.duration_s,
        "swing_radius_m": args.swing_radius_m,
        "turn_directions": list(args.turn_directions),
        "role_assignments": ROLE_ASSIGNMENTS,
        "repetitions": args.repetitions,
        "sample_count": len(samples),
        "stable_samples": sum(bool(item["stable"]) for item in samples),
        "groups_by_requested_angle": groups,
        "recommended_model_angle_allowlist_deg": allowlist,
        "allowlist_rule": (
            "retain an angle only when every geometry-matched direction/repetition "
            "completes without backend abort, floor/wall contact, robot fall, wrong-sign "
            "yaw, or missing final bilateral support, and actual directional payload yaw "
            "stays within 0.5x to 1.5x of the requested angle; opposite role assignments "
            "remain diagnostic evidence"
        ),
    }


def _wait_for_loaded_state(path: Path, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    last = ""
    stable_started: float | None = None
    while time.monotonic() < deadline:
        status = read_payload_runtime_status(path, max_age_s=1.0)
        last = status.describe()
        valid = (
            status.available
            and status.fresh
            and status.active
            and not status.floor_contact
            and status.min_robot_pelvis_height_m >= 0.50
            and {"front", "back"}.issubset(status.supporting_robots)
        )
        if valid:
            stable_started = stable_started or time.monotonic()
            if time.monotonic() - stable_started >= 0.5:
                return _read_json(path)
        else:
            stable_started = None
        time.sleep(0.05)
    raise RuntimeError(f"loaded payload state did not stabilize: {last}")


def _wait_for_checkpoint_release(
    path: Path,
    *,
    checkpoint_id: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        state = _read_json(path)
        checkpoint = state.get("checkpoint", {}) if isinstance(state, dict) else {}
        if (
            isinstance(checkpoint, dict)
            and str(checkpoint.get("checkpoint_id", "")) == checkpoint_id
            and not bool(checkpoint.get("resync_active", False))
        ):
            return
        time.sleep(0.05)
    raise RuntimeError(f"checkpoint {checkpoint_id!r} did not release after restore")


def _sample_runtime_state(state: dict[str, Any]) -> dict[str, Any]:
    payload = state.get("payload", {}) if isinstance(state.get("payload"), dict) else {}
    contacts = (
        payload.get("contacts", {}) if isinstance(payload.get("contacts"), dict) else {}
    )
    forces = contacts.get("support_forces", {})
    forces = forces if isinstance(forces, dict) else {}
    by_robot = forces.get("normal_force_n_by_robot", {})
    by_robot = by_robot if isinstance(by_robot, dict) else {}
    robots = state.get("robots", {}) if isinstance(state.get("robots"), dict) else {}
    pelvis = [
        float(values.get("pelvis_height_m", values.get("z", 0.0)))
        for values in robots.values()
        if isinstance(values, dict)
    ]
    supporting = {str(value) for value in contacts.get("support_robots", ())}
    return {
        "time_unix": float(state.get("time_unix", 0.0)),
        "front_support_force_n": float(by_robot.get("front", 0.0)),
        "back_support_force_n": float(by_robot.get("back", 0.0)),
        "support_force_balance_ratio": float(forces.get("balance_ratio", 0.0)),
        "bilateral_support": {"front", "back"}.issubset(supporting),
        "min_robot_pelvis_height_m": min(pelvis) if pelvis else 0.0,
        "payload_tilt_deg": float(payload.get("tilt_deg", 0.0)),
        "wall_contact": bool(contacts.get("walls", ())),
        "floor_contact": bool(contacts.get("floor", False)),
    }


def _write_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_calls": 0,
        "source_checkpoint_id": args.source_checkpoint_id,
        "baseline_checkpoint_id": args.baseline_checkpoint_id,
        "angles_deg": list(args.angles_deg),
        "turn_directions": list(args.turn_directions),
        "role_assignments": ROLE_ASSIGNMENTS,
        "repetitions": args.repetitions,
        "duration_s": args.duration_s,
        "swing_radius_m": args.swing_radius_m,
        "payload_safety_profile": args.payload_safety_profile,
        "scene_usda": str(args.scene_usda),
    }


def _available_checkpoint_ids(state: dict[str, Any]) -> tuple[str, ...]:
    checkpoint = state.get("checkpoint", {}) if isinstance(state, dict) else {}
    values = checkpoint.get("available_ids", ()) if isinstance(checkpoint, dict) else ()
    return tuple(str(value) for value in values)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_yaw(state: dict[str, Any]) -> float:
    payload = state.get("payload", {}) if isinstance(state.get("payload"), dict) else {}
    return float(payload.get("yaw_rad", 0.0))


def _wrapped_angle(value: float) -> float:
    return (float(value) + pi) % (2.0 * pi) - pi


def _robot_displacement(
    before: dict[str, Any], after: dict[str, Any], robot_id: str
) -> float:
    before_robot = before.get("robots", {}).get(robot_id, {})
    after_robot = after.get("robots", {}).get(robot_id, {})
    return hypot(
        float(after_robot.get("x", 0.0)) - float(before_robot.get("x", 0.0)),
        float(after_robot.get("y", 0.0)) - float(before_robot.get("y", 0.0)),
    )


def _supporting_robots(state: dict[str, Any]) -> set[str]:
    payload = state.get("payload", {}) if isinstance(state.get("payload"), dict) else {}
    contacts = (
        payload.get("contacts", {}) if isinstance(payload.get("contacts"), dict) else {}
    )
    return {str(value) for value in contacts.get("support_robots", ())}


def _compact_pose(state: dict[str, Any]) -> dict[str, Any]:
    payload = state.get("payload", {}) if isinstance(state.get("payload"), dict) else {}
    robots = state.get("robots", {}) if isinstance(state.get("robots"), dict) else {}
    return {
        "payload": {
            name: payload.get(name) for name in ("x", "y", "z", "yaw_rad", "tilt_deg")
        },
        "robots": {
            robot_id: {
                name: values.get(name)
                for name in ("x", "y", "z", "yaw_rad", "pelvis_height_m")
            }
            for robot_id, values in robots.items()
            if isinstance(values, dict)
        },
    }


def _turn_clearance(report: dict[str, Any], direction: str) -> float | None:
    record = _turn_clearance_record(report, direction)
    try:
        return float(record["clearance_m"])
    except (KeyError, TypeError, ValueError):
        return None


def _turn_clearance_record(report: dict[str, Any], direction: str) -> dict[str, Any]:
    candidates = report.get("turn_candidates", {}) if isinstance(report, dict) else {}
    record = candidates.get(direction, {}) if isinstance(candidates, dict) else {}
    return dict(record) if isinstance(record, dict) else {}


if __name__ == "__main__":
    main()
