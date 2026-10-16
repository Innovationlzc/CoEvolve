#!/usr/bin/env python3
"""Publish one static SONIC skill to one or two official deploy backends."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from dataclasses import replace
from math import cos, pi, sin
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.sonic.commands import SonicCommand, SonicLocomotionMode  # noqa: E402
from praxis_coop.sonic.skill_presets import (  # noqa: E402
    STATIC_SONIC_SKILLS,
    MODE_BY_NAME,
    SonicStaticSkillBuildConfig,
    build_static_sonic_skill_stream,
    list_static_sonic_skills,
    skill_requires_upper_body_lock,
)
from praxis_coop.sonic.runtime_state import (  # noqa: E402
    MIN_INITIAL_PAYLOAD_SUPPORT_MARGIN_M,
    PayloadRuntimeStatus,
    compute_runtime_motion_metrics,
    read_payload_runtime_status,
    read_runtime_motion_snapshot,
    wait_for_payload_on_target,
    wait_for_payload_recovery_ready,
    wait_for_payload_release_ready,
    wait_for_payload_supported,
    wait_for_upper_body_pose_transition,
)
from praxis_coop.sonic.safety import (  # noqa: E402
    CARRY_SAFETY_PROFILES,
    PAYLOAD_TOP_FACE_UP_LIMIT_DEG,
    TASK_CARRY_SAFETY,
    get_carry_safety_profile,
)
from praxis_coop.sonic.team_sync import (  # noqa: E402
    FormationSyncConfig,
    FormationSyncController,
)
from praxis_coop.sonic.stationary_hold import (  # noqa: E402
    PLACE_CROUCH_STATIONARY_LOCK_S,
    PLACE_CROUCH_STATIONARY_SETTLE_S,
    StationaryPlanarHold,
)
from praxis_coop.sonic.upper_body import (  # noqa: E402
    DEFAULT_CARRY_READY_POSE_NAME,
    PAYLOAD_RECOVERY_OBSERVATION_S,
    PAYLOAD_RECOVERY_POSE_NAME,
)
from praxis_coop.sonic.zmq_publisher import (  # noqa: E402
    SonicZmqConfig,
    SonicZmqPublisher,
    send_synchronized_planner_commands,
)


DEFAULT_PORTS = {
    "g1": 5556,
    "front": 5556,
    "back": 5558,
}

PAYLOAD_TRANSPORT_SKILLS = frozenset(
    {
        "carry_forward",
        "carry_backward",
        "carry_lateral_left",
        "carry_lateral_right",
        "carry_turn_in_place",
    }
)
PAYLOAD_TRANSLATION_SKILLS = frozenset(
    {"carry_forward", "carry_backward", "carry_lateral_left", "carry_lateral_right"}
)
PAYLOAD_PROTECTED_SKILLS = PAYLOAD_TRANSPORT_SKILLS | frozenset(
    {"carry_recover_payload_inward"}
)
_PROFILE_SPEED_PLATEAUS = 3


def _effective_post_skill_idle_seconds(args: argparse.Namespace) -> float:
    requested = max(0.0, float(args.post_skill_idle_seconds))
    if args.skill == "carry_recover_payload_inward":
        return max(requested, PAYLOAD_RECOVERY_OBSERVATION_S)
    return requested


def _wait_for_pre_transport_support(args: argparse.Namespace) -> PayloadRuntimeStatus:
    """Run the final four-palm gate after planner priming and pre-stand."""

    return wait_for_payload_supported(
        args.runtime_state_file,
        timeout_s=args.payload_verify_timeout_s,
        stable_s=args.payload_support_stable_s,
        max_age_s=args.runtime_state_max_age_s,
        min_z_m=args.payload_min_z,
        max_support_center_xy_error_m=(
            args.payload_safety_max_support_center_error_m
        ),
        require_all_support_pads=args.payload_require_all_support_pads,
        allow_non_support_robot_contacts=args.payload_safety_profile == "task",
        safety_profile=args.payload_safety_profile,
    )


def _payload_is_already_supported(
    status: PayloadRuntimeStatus,
    args: argparse.Namespace,
) -> bool:
    """Use the selected verifier profile for idempotent payload recovery."""

    return status.payload_supported(
        min_z_m=args.payload_min_z,
        require_all_support_pads=args.payload_require_all_support_pads,
        min_support_polygon_com_margin_m=MIN_INITIAL_PAYLOAD_SUPPORT_MARGIN_M,
        allow_non_support_robot_contacts=args.payload_safety_profile == "task",
        safety_profile=args.payload_safety_profile,
    )


def main() -> None:
    args = parse_args()
    post_skill_idle_seconds = _effective_post_skill_idle_seconds(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = _HeartbeatWriter(args.heartbeat_file)
    robot_ids = tuple(args.robots)
    ports = _resolve_ports(args, robot_ids)
    mode_name = None if args.mode == "skill_default" else args.mode

    stream = build_static_sonic_skill_stream(
        SonicStaticSkillBuildConfig(
            skill_name=args.skill,
            robot_ids=robot_ids,
            facing_xy=(args.facing_x, args.facing_y),
            movement_xy=None if args.movement_x is None or args.movement_y is None else (args.movement_x, args.movement_y),
            distance_m=args.distance_m,
            speed_mps=args.speed_mps,
            speed_mps_by_robot=tuple(
                (robot_id, speed)
                for robot_id, speed in (
                    ("front", args.front_speed_mps),
                    ("back", args.back_speed_mps),
                )
                if speed is not None and robot_id in robot_ids
            ),
            transport_ramp_seconds=args.transport_ramp_seconds,
            minimum_motion_duration_s=args.minimum_motion_duration_s,
            mode_name=mode_name,
            height=args.height,
            yaw_deg=args.yaw_deg,
            yaw_rate_degps=args.yaw_rate_degps,
            turn_segment_deg=args.turn_segment_deg,
            turn_command_scale=args.turn_command_scale,
            place_start_height=args.place_start_height,
            place_height=args.place_height,
            place_ramp_seconds=args.place_ramp_seconds,
            place_hold_seconds=args.place_hold_seconds,
            place_stationary_lock_seconds=args.place_stationary_lock_seconds,
            place_stationary_settle_seconds=args.place_stationary_settle_seconds,
            place_rise=args.place_rise,
            release_ramp_seconds=args.release_ramp_seconds,
            release_hold_seconds=args.release_hold_seconds,
            recovery_ramp_seconds=args.recovery_ramp_seconds,
            recovery_hold_seconds=args.recovery_hold_seconds,
            sync_group=args.sync_group,
        )
    )
    turn_state_file = _resolve_turn_state_file(args)
    stream, turn_state = _prepare_turn_heading_stream(
        stream,
        skill_name=args.skill,
        state_file=turn_state_file,
        reset_state=args.reset_turn_state,
        heading_mode=args.turn_heading_mode,
    )
    runtime_event_payload = _runtime_event_payload(args, started_at=time.time())
    publishers = {
        robot_id: SonicZmqPublisher(
            SonicZmqConfig(
                bind_host=args.bind_host,
                port=ports[robot_id],
                startup_delay_s=args.startup_delay,
                publish_period_s=1.0 / args.publish_hz,
                realtime=True,
                verbose=args.verbose_zmq,
            )
        )
        for robot_id in robot_ids
    }
    started_at = time.time()
    counts: list[dict[str, object]] = []
    payload_initialization: dict[str, object] | None = None
    payload_release: dict[str, object] | None = None
    payload_recovery_gate: dict[str, object] | None = None
    upper_body_recovery: dict[str, object] | None = None
    payload_event_dispatched = False
    execution_error: Exception | None = None
    motion_before: dict[str, object] | None = None
    motion_after: dict[str, object] | None = None
    motion_capture_errors: list[str] = []
    safety_monitor = None
    if (
        args.skill in PAYLOAD_PROTECTED_SKILLS
        and args.runtime_state_file is not None
        and not args.no_payload_safety_monitor
    ):
        safety_monitor = _PayloadSafetyMonitor(
            args.runtime_state_file,
            max_age_s=args.runtime_state_max_age_s,
            min_payload_z_m=args.payload_safety_min_z,
            min_robot_pelvis_height_m=args.payload_safety_min_pelvis_height,
            min_robot_xy_separation_m=args.payload_safety_min_robot_separation,
            max_tilt_deg=args.payload_safety_max_tilt_deg,
            hard_tilt_deg=args.payload_safety_hard_tilt_deg,
            tilt_grace_s=args.payload_tilt_grace_s,
            contact_loss_grace_s=args.payload_contact_loss_grace_s,
            max_support_center_xy_error_m=args.payload_safety_max_support_center_error_m,
            min_support_force_per_robot_n=args.payload_support_min_force_n,
            min_support_force_balance_ratio=args.payload_support_min_balance_ratio,
            force_imbalance_grace_s=args.payload_force_imbalance_grace_s,
            max_top_surface_z_range_m=args.payload_safety_max_top_surface_range_m,
            wall_contact_is_hard_failure=args.payload_wall_contact_is_hard_failure,
            support_center_is_hard_failure=args.payload_support_center_is_hard_failure,
            strict_geometry_checks=args.payload_safety_profile == "strict",
        )
    team_sync_controller = None
    if (
        args.skill in PAYLOAD_TRANSLATION_SKILLS
        and args.runtime_state_file is not None
        and set(robot_ids) == {"front", "back"}
        and not args.no_team_sync
    ):
        team_sync_controller = FormationSyncController(
            args.runtime_state_file,
            FormationSyncConfig(
                max_state_age_s=args.runtime_state_max_age_s,
                active_catchup=args.team_sync_active_catchup,
                longitudinal_kp_s=args.team_sync_longitudinal_kp,
                cross_track_kp_s=args.team_sync_cross_track_kp,
                payload_kp_s=args.team_sync_payload_kp,
                error_filter_alpha=args.team_sync_error_filter_alpha,
                pause_leader_error_m=args.team_sync_pause_leader_error_m,
                resume_error_m=args.team_sync_resume_error_m,
                lagging_speed_boost_mps=args.team_sync_lagging_boost_mps,
                max_correction_mps=args.team_sync_max_correction_mps,
                min_active_speed_ratio=args.team_sync_min_active_speed_ratio,
                hard_desync_m=args.team_sync_hard_desync_m,
            ),
        )
    planar_hold = None
    if (
        args.skill == "carry_place_crouch"
        and args.runtime_state_file is not None
        and args.place_crouch_planar_hold
    ):
        planar_hold = StationaryPlanarHold(
            args.runtime_state_file,
            max_state_age_s=args.runtime_state_max_age_s,
        )
    pre_stand_height = args.height
    if args.skill == "carry_place_crouch" and args.height < 0.0:
        pre_stand_height = args.place_start_height

    try:
        print(
            f"[sonic-static-skill] skill={args.skill} robots={','.join(robot_ids)} "
            f"ports={ports} requires_upper_body_lock={skill_requires_upper_body_lock(args.skill)}",
            flush=True,
        )
        if skill_requires_upper_body_lock(args.skill):
            print(
                "[sonic-static-skill] keep Term1 motor-side lock enabled, usually "
                f"--upper-body-lock-pose {DEFAULT_CARRY_READY_POSE_NAME} with "
                "--upper-body-lock-palm-tilt-deg 60; "
                "this publisher does not send planner-side arm targets",
                flush=True,
            )
        if team_sync_controller is not None:
            hard_guard = (
                "disabled"
                if args.team_sync_hard_desync_m is None
                else f"{args.team_sync_hard_desync_m:.3f}m"
            )
            print(
                "[sonic-static-skill] live dual-robot formation telemetry enabled "
                f"mode={'smooth_longitudinal_catchup' if args.team_sync_active_catchup else 'equal_command'} "
                f"pause={args.team_sync_pause_leader_error_m:.3f}m "
                f"resume={args.team_sync_resume_error_m:.3f}m "
                f"hard_guard={hard_guard}",
                flush=True,
            )
        if planar_hold is not None:
            print(
                "[sonic-static-skill] place_crouch stationary planar hold enabled "
                f"enter={planar_hold.enter_deadband_m:.3f}m "
                f"exit={planar_hold.exit_deadband_m:.3f}m "
                f"max_speed={planar_hold.max_speed_mps:.3f}m/s",
                flush=True,
            )
        pre_stand_commands = {
            robot_id: _idle_command(
                robot_id=robot_id,
                facing=(args.facing_x, args.facing_y, 0.0),
                height=pre_stand_height,
                duration_s=max(args.stand_seconds, args.start_pulse_seconds),
                phase_id="static_skill_pre_stand",
                sync_group=args.sync_group,
            )
            for robot_id in robot_ids
        }

        prime_idle = not args.no_prime_idle_during_start_pulses
        print(
            f"[sonic-static-skill] sending start pulses for {args.start_pulse_seconds:.1f}s "
            f"(prime_idle={prime_idle})",
            flush=True,
        )
        counts.append(
            {
                "phase": "start_pulses",
                "messages_per_robot": _publish_start_pulses(
                    publishers,
                    duration_s=args.start_pulse_seconds,
                    publish_hz=args.start_pulse_hz,
                    heartbeat=heartbeat,
                    idle_commands=pre_stand_commands if prime_idle else None,
                ),
            }
        )

        if args.stand_seconds > 0.0:
            print(f"[sonic-static-skill] pre-stand for {args.stand_seconds:.1f}s", flush=True)
            counts.append(
                {
                    "phase": "pre_stand",
                    "messages_per_robot": _publish_command_group(
                        publishers,
                        pre_stand_commands,
                        duration_s=args.stand_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                        turn_heading_mode=args.turn_heading_mode,
                        runtime_event_payload={},
                    ),
                }
            )

        payload_already_supported = False
        if args.skill == "place_payload_on_arms":
            if args.runtime_state_file is None:
                raise RuntimeError(
                    "place_payload_on_arms requires --runtime-state-file (normally the file written by Term1)"
                )
            print(
                "[sonic-static-skill] waiting for both robots to be dropped, upright, "
                "stationary, and at the carry-ready arm pose",
                flush=True,
            )
            ready_status, ready_count = _wait_for_payload_readiness(
                publishers,
                pre_stand_commands,
                heartbeat=heartbeat,
                runtime_state_file=args.runtime_state_file,
                timeout_s=args.payload_ready_timeout_s,
                stable_s=args.payload_ready_stable_s,
                max_age_s=args.runtime_state_max_age_s,
                publish_hz=args.publish_hz,
            )
            counts.append(
                {
                    "phase": "payload_initialization_wait_ready",
                    "messages_per_robot": ready_count,
                }
            )
            payload_force = bool(runtime_event_payload.get("payload_force", False))
            payload_already_supported = _payload_is_already_supported(
                ready_status,
                args,
            )
            if payload_already_supported:
                print(
                    "[sonic-static-skill] payload is already stably supported; "
                    "keeping the robots idle without teleporting it again, even when "
                    "force=True was requested by recovery",
                    flush=True,
                )
            elif payload_force:
                print(
                    "[sonic-static-skill] guarded payload force requested; "
                    "replaying the placement event after the readiness gate",
                    flush=True,
                )
            elif ready_status.active:
                if args.no_payload_auto_reset:
                    raise RuntimeError(
                        "payload is active but not supported; restart Term1 or omit "
                        "--no-payload-auto-reset to re-place only the payload"
                    )
                runtime_event_payload["payload_force"] = True
                print(
                    "[sonic-static-skill] payload is active but unsupported; "
                    "enabling one guarded payload-only re-placement",
                    flush=True,
                )

        if args.skill == "release_payload":
            if args.runtime_state_file is None:
                raise RuntimeError("release_payload requires --runtime-state-file")
            print(
                "[sonic-static-skill] waiting for a loaded payload above the physical target",
                flush=True,
            )
            release_ready = wait_for_payload_release_ready(
                args.runtime_state_file,
                timeout_s=args.payload_ready_timeout_s,
                stable_s=args.payload_ready_stable_s,
                max_age_s=args.runtime_state_max_age_s,
                target_xy_tolerance_m=args.payload_target_xy_tolerance_m,
                max_drop_to_target_m=args.payload_release_max_drop_m,
            )
            payload_release = {
                "release_gate": "passed",
                "payload_before": release_ready.get("payload", {}),
                "target": release_ready.get("target", {}),
            }

        if safety_monitor is not None:
            supported = _wait_for_pre_transport_support(args)
            safety_monitor.observe(supported)
            print(
                "[sonic-static-skill] pre-transport physical guard passed: "
                + supported.describe(),
                flush=True,
            )

        if args.skill == "carry_recover_payload_inward":
            assert args.runtime_state_file is not None
            recovery_ready = wait_for_payload_recovery_ready(
                args.runtime_state_file,
                timeout_s=args.payload_ready_timeout_s,
                stable_s=min(0.5, args.payload_ready_stable_s),
                max_age_s=args.runtime_state_max_age_s,
                minimum_outward_offset_m=(
                    TASK_CARRY_SAFETY.warning_payload_formation_slip_m
                ),
            )
            recovery_payload = recovery_ready.get("payload", {})
            recovery_lock = recovery_ready.get("upper_body_pose_lock", {})
            payload_recovery_gate = {
                "status": "passed",
                "pose_before": (
                    recovery_lock.get("pose_name")
                    if isinstance(recovery_lock, dict)
                    else ""
                ),
                "outward_offset_before_m": (
                    recovery_payload.get(
                        "support_center_signed_longitudinal_offset_m"
                    )
                    if isinstance(recovery_payload, dict)
                    else None
                ),
                "minimum_outward_offset_m": (
                    TASK_CARRY_SAFETY.warning_payload_formation_slip_m
                ),
            }
            print(
                "[sonic-static-skill] payload recovery gate passed: "
                f"pose={payload_recovery_gate['pose_before']} "
                f"outward_offset={payload_recovery_gate['outward_offset_before_m']}m",
                flush=True,
            )

        if args.skill in PAYLOAD_TRANSLATION_SKILLS and args.runtime_state_file is not None:
            motion_before = read_runtime_motion_snapshot(args.runtime_state_file)

        if payload_already_supported:
            counts.append(
                {
                    "phase": "place_payload_on_arms_already_supported_hold",
                    "messages_per_robot": _publish_command_group(
                        publishers,
                        pre_stand_commands,
                        duration_s=args.place_hold_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                        turn_heading_mode=args.turn_heading_mode,
                        runtime_event_payload={},
                    ),
                }
            )
        else:
            print(
                f"[sonic-static-skill] executing {len(stream.commands)} command(s), "
                f"duration={stream.duration_s:.1f}s",
                flush=True,
            )
            counts.extend(
                _publish_stream_groups(
                    publishers,
                    stream.sorted().commands,
                    publish_hz=args.publish_hz,
                    heartbeat=heartbeat,
                    turn_heading_mode=args.turn_heading_mode,
                    runtime_event_payload=runtime_event_payload,
                    safety_monitor=safety_monitor,
                    team_sync_controller=team_sync_controller,
                    planar_hold=planar_hold,
                )
            )
            payload_event_dispatched = args.skill in {
                "place_payload_on_arms",
                "release_payload",
                "carry_recover_payload_inward",
            }

        if args.skill == "carry_recover_payload_inward":
            assert args.runtime_state_file is not None
            event_id = str(runtime_event_payload["upper_body_event_id"])
            recovery_state = wait_for_upper_body_pose_transition(
                args.runtime_state_file,
                pose_name=PAYLOAD_RECOVERY_POSE_NAME,
                event_id=event_id,
                timeout_s=args.payload_verify_timeout_s,
                stable_s=min(0.5, args.payload_support_stable_s),
                max_age_s=args.runtime_state_max_age_s,
            )
            upper_body_recovery = dict(
                recovery_state.get("upper_body_pose_lock", {})
            )
            print(
                "[sonic-static-skill] synchronized payload recovery verified: "
                f"pose={upper_body_recovery.get('pose_name')} "
                f"event_id={event_id}; waist/legs unchanged",
                flush=True,
            )

        if safety_monitor is not None:
            post_idle_commands = {
                robot_id: _idle_command(
                    robot_id=robot_id,
                    facing=(args.facing_x, args.facing_y, 0.0),
                    height=args.height,
                    duration_s=post_skill_idle_seconds,
                    phase_id=f"{args.skill}.post_skill_idle",
                    sync_group=args.sync_group,
                )
                for robot_id in robot_ids
            }
            counts.append(
                {
                    "phase": f"{args.skill}.post_skill_idle",
                    "messages_per_robot": _publish_command_group(
                        publishers,
                        post_idle_commands,
                        duration_s=post_skill_idle_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                        turn_heading_mode=args.turn_heading_mode,
                        runtime_event_payload={},
                        safety_monitor=safety_monitor,
                    ),
                }
            )
            supported = wait_for_payload_supported(
                args.runtime_state_file,
                timeout_s=args.payload_verify_timeout_s,
                stable_s=args.payload_support_stable_s,
                max_age_s=args.runtime_state_max_age_s,
                min_z_m=args.payload_min_z,
                allow_non_support_robot_contacts=args.payload_safety_profile == "task",
                safety_profile=args.payload_safety_profile,
            )
            safety_monitor.observe(supported)
            if args.skill == "carry_recover_payload_inward" and upper_body_recovery is not None:
                upper_body_recovery["stability_observation_s"] = post_skill_idle_seconds
                upper_body_recovery["payload_after_observation"] = _payload_status_json(
                    supported,
                    args,
                )
            print(
                "[sonic-static-skill] post-transport physical guard passed: "
                + supported.describe(),
                flush=True,
            )

        if args.skill == "place_payload_on_arms":
            assert args.runtime_state_file is not None
            supported = wait_for_payload_supported(
                args.runtime_state_file,
                timeout_s=args.payload_verify_timeout_s,
                stable_s=args.payload_support_stable_s,
                max_age_s=args.runtime_state_max_age_s,
                min_z_m=args.payload_min_z,
                require_all_support_pads=args.payload_require_all_support_pads,
                min_support_polygon_com_margin_m=(
                    MIN_INITIAL_PAYLOAD_SUPPORT_MARGIN_M
                ),
                allow_non_support_robot_contacts=(
                    args.payload_safety_profile == "task"
                ),
                safety_profile=args.payload_safety_profile,
            )
            payload_initialization = _payload_status_json(supported, args)
            payload_initialization["placement_action"] = (
                "kept_existing_payload" if payload_already_supported else "activated_deferred_payload"
            )
            print(
                "[sonic-static-skill] payload initialization verified continuously: "
                + supported.describe(),
                flush=True,
            )

        if args.skill == "release_payload":
            assert args.runtime_state_file is not None
            target_state = wait_for_payload_on_target(
                args.runtime_state_file,
                timeout_s=args.payload_verify_timeout_s,
                stable_s=args.payload_support_stable_s,
                max_age_s=args.runtime_state_max_age_s,
                target_xy_tolerance_m=args.payload_target_xy_tolerance_m,
                target_z_tolerance_m=args.payload_target_z_tolerance_m,
                require_released=True,
                allow_robot_contact=True,
            )
            payload_release = {
                **(payload_release or {}),
                "verified": True,
                "payload_after": target_state.get("payload", {}),
                "target": target_state.get("target", {}),
            }
            print(
                "[sonic-static-skill] payload release verified: target geometry carries "
                "the settled box; residual arm contact may end during crouched retreat",
                flush=True,
            )

        if args.final_idle_seconds > 0.0:
            final_commands = {
                robot_id: _idle_command(
                    robot_id=robot_id,
                    facing=(args.facing_x, args.facing_y, 0.0),
                    height=args.height,
                    duration_s=args.final_idle_seconds,
                    phase_id="static_skill_final_idle",
                    sync_group=args.sync_group,
                )
                for robot_id in robot_ids
            }
            print(f"[sonic-static-skill] final idle for {args.final_idle_seconds:.1f}s", flush=True)
            counts.append(
                {
                    "phase": "final_idle",
                    "messages_per_robot": _publish_command_group(
                        publishers,
                        final_commands,
                        duration_s=args.final_idle_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                        turn_heading_mode=args.turn_heading_mode,
                        runtime_event_payload={},
                        safety_monitor=safety_monitor,
                    ),
                }
            )

        if not args.no_stop:
            for publisher in publishers.values():
                publisher.send_stop()
    except Exception as exc:  # noqa: BLE001 - persist physical failure evidence before propagating
        execution_error = exc
        print(f"[sonic-static-skill] failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        for publisher in publishers.values():
            publisher.close()

    if motion_before is not None and args.runtime_state_file is not None:
        try:
            motion_after = read_runtime_motion_snapshot(args.runtime_state_file)
        except RuntimeError as exc:
            motion_capture_errors.append(str(exc))
    physical_motion = _physical_motion_report(
        stream,
        before=motion_before,
        after=motion_after,
        capture_errors=motion_capture_errors,
    )
    metrics = physical_motion.get("metrics")
    if isinstance(metrics, dict):
        payload_metrics = metrics.get("payload", {})
        team_metrics = metrics.get("team_mean", {})
        support_slip = metrics.get("payload_relative_to_support_target", {})
        print(
            "[sonic-static-skill] measured MuJoCo motion "
            f"payload_progress={float(payload_metrics.get('target_direction_progress_m', 0.0)):.4f}m "
            f"team_progress={float(team_metrics.get('target_direction_progress_m', 0.0)):.4f}m "
            f"robot_gap={float(metrics.get('robot_target_progress_gap_m', 0.0)):.4f}m "
            f"support_slip={float(support_slip.get('net_displacement_m', 0.0)):.4f}m",
            flush=True,
        )

    result = {
        "status": (
            f"failed_{type(execution_error).__name__}"
            if execution_error is not None
            else (
                "payload_initialization_verified"
                if payload_initialization is not None and bool(payload_initialization.get("verified", False))
                else "payload_release_verified"
                if payload_release is not None and bool(payload_release.get("verified", False))
                else "upper_body_recovery_verified"
                if upper_body_recovery is not None
                else "static_sonic_skill_commands_sent_manual_visual_verification_required"
            )
        ),
        "error": "" if execution_error is None else f"{type(execution_error).__name__}: {execution_error}",
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "skill": args.skill,
        "robots": list(robot_ids),
        "ports": ports,
        "requires_upper_body_lock": skill_requires_upper_body_lock(args.skill),
        "heartbeat_file": str(args.heartbeat_file) if args.heartbeat_file is not None else None,
        "counts": counts,
        "commands": stream.to_json(),
        "turn_state_file": str(turn_state_file) if turn_state_file is not None else None,
        "turn_state": turn_state,
        "runtime_event": runtime_event_payload if payload_event_dispatched else {},
        "runtime_event_dispatched": payload_event_dispatched,
        "payload_initialization": payload_initialization,
        "payload_release": payload_release,
        "payload_recovery_gate": payload_recovery_gate,
        "upper_body_recovery": upper_body_recovery,
        "payload_recovery_observation_s": (
            post_skill_idle_seconds
            if args.skill == "carry_recover_payload_inward"
            else None
        ),
        "payload_safety_monitor": None if safety_monitor is None else safety_monitor.report(),
        "team_synchronization": (
            {"enabled": False}
            if team_sync_controller is None
            else team_sync_controller.report()
        ),
        "physical_motion": physical_motion,
        "success_criterion": (
            [
                "both official SONIC-controlled G1 robots are dropped, upright, and carry-pose ready",
                "the one deferred long_box is activated only after the live readiness gate passes",
                "front and back continuously support the payload above the minimum height",
                "the payload has no floor or wall contact",
            ]
            if args.skill == "place_payload_on_arms"
            else [
                "both robots retain continuous legal upper-limb payload support",
                f"Term1 reports completed pose {PAYLOAD_RECOVERY_POSE_NAME}",
                "only shoulder/elbow arm-lock targets change; waist and legs remain unchanged",
                "the payload stays above the floor and neither robot falls",
            ]
            if args.skill == "carry_recover_payload_inward"
            else [
                "official MuJoCo runtime remains upright after ElasticBand drop",
                "the selected robot(s) execute the requested skill under official SONIC control",
                "for carry_* skills, Term1 motor-side upper-body lock keeps arms/hands unchanged",
                "no root-pose proxy or Isaac fake success is used",
            ]
        ),
    }
    output_file = args.output_dir / f"{args.skill}_result.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[sonic-static-skill] wrote {output_file}", flush=True)
    if execution_error is not None:
        raise execution_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-skills", action="store_true", help="Print available static skills and exit.")
    parser.add_argument("--skill", choices=sorted(STATIC_SONIC_SKILLS), default="parallel_forward")
    parser.add_argument(
        "--robots",
        nargs="+",
        choices=sorted(DEFAULT_PORTS),
        default=["front", "back"],
        help="Use 'front back' for dual-G1, or one of 'front', 'back', 'g1' for single-robot publishing.",
    )
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--front-port", type=int, default=5556)
    parser.add_argument("--back-port", type=int, default=5558)
    parser.add_argument("--g1-port", type=int, default=5556)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--start-pulse-seconds", type=float, default=8.0)
    parser.add_argument("--start-pulse-hz", type=float, default=10.0)
    parser.add_argument(
        "--no-prime-idle-during-start-pulses",
        action="store_true",
        help=(
            "Only send start pulses during the startup window. By default, static skills also "
            "send an IDLE pre-stand planner command and heartbeat during startup so the guarded "
            "drop can happen as soon as lowcmd is ready."
        ),
    )
    parser.add_argument("--stand-seconds", type=float, default=5.0)
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--speed-mps", type=float, default=None)
    parser.add_argument(
        "--front-speed-mps",
        type=float,
        default=None,
        help="Optional front-robot speed override while retaining the shared motion window.",
    )
    parser.add_argument(
        "--back-speed-mps",
        type=float,
        default=None,
        help="Optional back-robot speed override while retaining the shared motion window.",
    )
    parser.add_argument(
        "--transport-ramp-seconds",
        type=float,
        default=0.0,
        help=(
            "True IDLE settle before and after the fixed-speed move phase. The verified "
            "manual payload envelope uses 1.5 seconds."
        ),
    )
    parser.add_argument(
        "--minimum-motion-duration-s",
        type=float,
        default=0.4,
        help=(
            "Minimum active SONIC gait window for translation skills. This is a command-window "
            "guard, not a claim that target_vel equals measured metres per second."
        ),
    )
    parser.add_argument("--mode", choices=("skill_default", *sorted(MODE_BY_NAME)), default="skill_default")
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--facing-x", type=float, default=1.0)
    parser.add_argument("--facing-y", type=float, default=0.0)
    parser.add_argument(
        "--movement-x",
        type=float,
        default=None,
        help="Optional explicit movement x direction. Mainly used for lateral skills while preserving planner facing.",
    )
    parser.add_argument(
        "--movement-y",
        type=float,
        default=None,
        help="Optional explicit movement y direction. Must be provided with --movement-x.",
    )
    parser.add_argument("--yaw-deg", type=float, default=90.0)
    parser.add_argument("--yaw-rate-degps", type=float, default=8.0)
    parser.add_argument("--turn-segment-deg", type=float, default=5.0)
    parser.add_argument(
        "--turn-command-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the requested yaw before sending heading increments. "
            "Use this to calibrate visual body yaw against SONIC heading state."
        ),
    )
    parser.add_argument(
        "--turn-heading-mode",
        choices=("incremental", "absolute", "absolute_state"),
        default="incremental",
        help=(
            "How yaw is sent to SONIC. 'incremental' mirrors keyboard Q/E and is the stable default. "
            "'absolute' and 'absolute_state' use the older absolute delta_heading path."
        ),
    )
    parser.add_argument(
        "--turn-state-file",
        type=Path,
        default=None,
        help=(
            "Persistent cumulative heading state for repeated turn skills. "
            "Defaults to <heartbeat-dir>/turn_heading_state.json when --heartbeat-file is set."
        ),
    )
    parser.add_argument(
        "--reset-turn-state",
        action="store_true",
        help="Reset the persistent cumulative heading state before publishing this turn.",
    )
    parser.add_argument(
        "--turn-absolute",
        action="store_true",
        help="Deprecated alias for --turn-heading-mode absolute.",
    )
    parser.add_argument("--place-start-height", type=float, default=0.80)
    parser.add_argument("--place-height", type=float, default=0.55)
    parser.add_argument("--place-ramp-seconds", type=float, default=5.0)
    parser.add_argument("--place-hold-seconds", type=float, default=4.0)
    parser.add_argument(
        "--place-stationary-lock-seconds",
        type=float,
        default=PLACE_CROUCH_STATIONARY_LOCK_S,
        help="Idle lock at the standing height before place_crouch lowers.",
    )
    parser.add_argument(
        "--place-stationary-settle-seconds",
        type=float,
        default=PLACE_CROUCH_STATIONARY_SETTLE_S,
        help="Idle settle at the placed height after place_crouch hold.",
    )
    parser.add_argument(
        "--place-crouch-planar-hold",
        action="store_true",
        help=(
            "Optional weak planner XY hold during place_crouch. Default off: the "
            "in-place contract is enforced by a vertical motor-lock squat, and "
            "SLOW_WALK cannot override locked legs during the lower/hold phases."
        ),
    )
    parser.add_argument("--place-rise", action="store_true")
    parser.add_argument("--release-ramp-seconds", type=float, default=4.0)
    parser.add_argument("--release-hold-seconds", type=float, default=2.0)
    parser.add_argument("--recovery-ramp-seconds", type=float, default=6.0)
    parser.add_argument("--recovery-hold-seconds", type=float, default=2.0)
    parser.add_argument("--upper-body-event-id", default="")
    parser.add_argument("--payload-release-max-drop-m", type=float, default=0.45)
    parser.add_argument(
        "--payload-target-xy-tolerance-m",
        type=float,
        default=None,
        help=(
            "Optional diagnostic override. By default release readiness uses the "
            "physical target cylinder radius and final success uses stable target contact."
        ),
    )
    parser.add_argument(
        "--payload-target-z-tolerance-m",
        type=float,
        default=None,
        help=(
            "Optional diagnostic override. Final success normally uses stable physical "
            "target contact rather than a fixed center-height tolerance."
        ),
    )
    parser.add_argument("--payload-event-id", default="")
    parser.add_argument("--payload-x", type=float, default=None)
    parser.add_argument("--payload-y", type=float, default=None)
    parser.add_argument("--payload-z", type=float, default=None)
    parser.add_argument("--payload-yaw", type=float, default=None)
    parser.add_argument(
        "--payload-rearward-offset-m",
        type=float,
        default=None,
        help=(
            "Override Term1's live support-center rearward offset for this guarded "
            "placement event and future placements/checkpoints."
        ),
    )
    parser.add_argument(
        "--payload-force",
        action="store_true",
        help="Re-place the payload even if the runtime already marked it active.",
    )
    parser.add_argument(
        "--runtime-state-file",
        type=Path,
        default=None,
        help="Live runtime_state.json written by Term1; required for guarded payload initialization.",
    )
    parser.add_argument("--runtime-state-max-age-s", type=float, default=2.0)
    parser.add_argument("--payload-ready-timeout-s", type=float, default=45.0)
    parser.add_argument("--payload-ready-stable-s", type=float, default=1.0)
    parser.add_argument(
        "--payload-verify-timeout-s",
        type=float,
        default=15.0,
        help="Maximum event-driven wait for the free payload to settle into stable dual support.",
    )
    parser.add_argument("--payload-support-stable-s", type=float, default=1.0)
    parser.add_argument("--payload-min-z", type=float, default=0.80)
    parser.add_argument(
        "--payload-safety-profile",
        choices=tuple(CARRY_SAFETY_PROFILES),
        default="task",
        help="Select the complete physical verifier envelope; task is the experiment default.",
    )
    parser.add_argument("--payload-safety-min-z", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-safety-min-pelvis-height", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--payload-safety-min-robot-separation",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--payload-safety-max-tilt-deg",
        type=float,
        default=None,
        help="Soft in-motion payload tilt limit; brief gait transients use --payload-tilt-grace-s.",
    )
    parser.add_argument(
        "--payload-safety-hard-tilt-deg",
        type=float,
        default=None,
        help="Immediate payload tilt abort independent of the transient grace window.",
    )
    parser.add_argument(
        "--payload-tilt-grace-s",
        type=float,
        default=None,
        help="Maximum continuous time above the soft tilt limit before aborting.",
    )
    parser.add_argument(
        "--payload-safety-max-support-center-error-m",
        type=float,
        default=None,
        help="Abort carry motion when payload COM drifts this far from the live four-palm support target.",
    )
    parser.add_argument(
        "--no-team-sync",
        action="store_true",
        help="Disable live front/back root-pose feedback. Explicit ablation/debug only.",
    )
    parser.add_argument(
        "--team-sync-active-catchup",
        action="store_true",
        help=(
            "Experimentally issue different leader/lagger speeds. The verified default keeps "
            "both SONIC commands identical and uses formation feedback only as a hard guard."
        ),
    )
    parser.add_argument("--team-sync-longitudinal-kp", type=float, default=2.0)
    parser.add_argument("--team-sync-cross-track-kp", type=float, default=0.6)
    parser.add_argument("--team-sync-payload-kp", type=float, default=1.2)
    parser.add_argument(
        "--team-sync-error-filter-alpha",
        type=float,
        default=0.35,
        help="Low-rate root-error filter alpha; larger values react earlier to gait-onset desynchronization.",
    )
    parser.add_argument("--team-sync-pause-leader-error-m", type=float, default=0.008)
    parser.add_argument("--team-sync-resume-error-m", type=float, default=0.003)
    parser.add_argument("--team-sync-lagging-boost-mps", type=float, default=0.03)
    parser.add_argument("--team-sync-max-correction-mps", type=float, default=0.06)
    parser.add_argument("--team-sync-min-active-speed-ratio", type=float, default=0.0)
    parser.add_argument(
        "--team-sync-hard-desync-m",
        type=float,
        default=None,
        help="Stop both deploys before a root-formation split can pull the payload off its supports.",
    )
    parser.add_argument("--payload-contact-loss-grace-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--payload-support-min-force-n",
        type=float,
        default=None,
        help="Minimum measured palm-normal load per robot when MuJoCo force telemetry is available.",
    )
    parser.add_argument(
        "--payload-support-min-balance-ratio",
        type=float,
        default=None,
        help="Minimum min(front,back)/max(front,back) palm load ratio during transport.",
    )
    parser.add_argument(
        "--payload-force-imbalance-grace-s",
        type=float,
        default=None,
        help="Allow brief gait load transfer before treating one-sided support as physical failure.",
    )
    parser.add_argument("--post-skill-idle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--no-payload-safety-monitor",
        action="store_true",
        help="Disable live payload/robot safety aborts for carry skills. Debug/ablation only.",
    )
    parser.add_argument(
        "--no-payload-auto-reset",
        action="store_true",
        help="Fail instead of re-placing an already-active payload that is no longer supported.",
    )
    parser.add_argument("--final-idle-seconds", type=float, default=3600.0)
    parser.add_argument("--sync-group", default="static_sonic_skill")
    parser.add_argument("--no-stop", action="store_true")
    parser.add_argument("--verbose-zmq", action="store_true")
    parser.add_argument(
        "--heartbeat-file",
        type=Path,
        default=None,
        help="Continuously touch/write this file while planner commands are published.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "deployment" / "static_sonic_skills",
    )
    args = parser.parse_args()
    safety = get_carry_safety_profile(args.payload_safety_profile)
    args.payload_safety_min_z = (
        safety.min_payload_z_m
        if args.payload_safety_min_z is None
        else args.payload_safety_min_z
    )
    args.payload_safety_min_pelvis_height = (
        safety.min_robot_pelvis_height_m
        if args.payload_safety_min_pelvis_height is None
        else args.payload_safety_min_pelvis_height
    )
    args.payload_safety_min_robot_separation = (
        safety.min_robot_xy_separation_m
        if args.payload_safety_min_robot_separation is None
        else args.payload_safety_min_robot_separation
    )
    args.payload_safety_max_tilt_deg = (
        safety.settled_max_tilt_deg
        if args.payload_safety_max_tilt_deg is None
        else args.payload_safety_max_tilt_deg
    )
    args.payload_safety_hard_tilt_deg = (
        safety.hard_max_tilt_deg
        if args.payload_safety_hard_tilt_deg is None
        else args.payload_safety_hard_tilt_deg
    )
    args.payload_tilt_grace_s = (
        safety.support_loss_grace_s
        if args.payload_tilt_grace_s is None
        else args.payload_tilt_grace_s
    )
    args.payload_contact_loss_grace_s = (
        safety.support_loss_grace_s
        if args.payload_contact_loss_grace_s is None
        else args.payload_contact_loss_grace_s
    )
    args.payload_safety_max_support_center_error_m = (
        safety.max_support_center_error_m
        if args.payload_safety_max_support_center_error_m is None
        else args.payload_safety_max_support_center_error_m
    )
    args.payload_support_min_force_n = (
        safety.min_support_force_per_robot_n
        if args.payload_support_min_force_n is None
        else args.payload_support_min_force_n
    )
    args.payload_support_min_balance_ratio = (
        safety.min_support_force_balance_ratio
        if args.payload_support_min_balance_ratio is None
        else args.payload_support_min_balance_ratio
    )
    args.payload_force_imbalance_grace_s = (
        safety.support_loss_grace_s
        if args.payload_force_imbalance_grace_s is None
        else args.payload_force_imbalance_grace_s
    )
    args.team_sync_hard_desync_m = (
        safety.team_hard_desync_m
        if args.team_sync_hard_desync_m is None
        else args.team_sync_hard_desync_m
    )
    args.payload_safety_max_top_surface_range_m = safety.max_top_surface_z_range_m
    args.payload_wall_contact_is_hard_failure = safety.wall_contact_is_hard_failure
    args.payload_support_center_is_hard_failure = safety.name == "strict"
    args.payload_require_all_support_pads = safety.require_all_support_pads_when_settled
    if args.team_sync_resume_error_m < 0.0:
        parser.error("--team-sync-resume-error-m must be non-negative")
    if args.team_sync_pause_leader_error_m <= args.team_sync_resume_error_m:
        parser.error(
            "--team-sync-pause-leader-error-m must be greater than --team-sync-resume-error-m"
        )
    if (
        args.team_sync_hard_desync_m is not None
        and args.team_sync_hard_desync_m <= args.team_sync_pause_leader_error_m
    ):
        parser.error(
            "--team-sync-hard-desync-m must be greater than --team-sync-pause-leader-error-m"
        )
    if min(
        args.team_sync_longitudinal_kp,
        args.team_sync_cross_track_kp,
        args.team_sync_payload_kp,
        args.team_sync_lagging_boost_mps,
        args.team_sync_max_correction_mps,
    ) < 0.0:
        parser.error("team synchronization gains and speed limits must be non-negative")
    if args.payload_support_min_force_n < 0.0:
        parser.error("--payload-support-min-force-n must be non-negative")
    if not 0.0 <= args.team_sync_min_active_speed_ratio <= 1.0:
        parser.error("--team-sync-min-active-speed-ratio must be within [0, 1]")
    if not 0.0 <= args.team_sync_error_filter_alpha <= 1.0:
        parser.error("--team-sync-error-filter-alpha must be within [0, 1]")
    if args.payload_safety_max_tilt_deg < 0.0:
        parser.error("--payload-safety-max-tilt-deg must be non-negative")
    if args.payload_safety_hard_tilt_deg < args.payload_safety_max_tilt_deg:
        parser.error(
            "--payload-safety-hard-tilt-deg must be at least --payload-safety-max-tilt-deg"
        )
    if args.payload_tilt_grace_s < 0.0:
        parser.error("--payload-tilt-grace-s must be non-negative")
    if not 0.0 <= args.payload_support_min_balance_ratio <= 1.0:
        parser.error("--payload-support-min-balance-ratio must be within [0, 1]")
    if args.payload_force_imbalance_grace_s < 0.0:
        parser.error("--payload-force-imbalance-grace-s must be non-negative")
    if args.runtime_state_file is None and args.heartbeat_file is not None:
        args.runtime_state_file = args.heartbeat_file.parent / "runtime_state.json"
    if args.skill == "carry_recover_payload_inward":
        if len(args.robots) != 2 or set(args.robots) != {"front", "back"}:
            parser.error(
                "carry_recover_payload_inward requires exactly --robots front back"
            )
        if args.runtime_state_file is None:
            parser.error(
                "carry_recover_payload_inward requires --runtime-state-file"
            )
    if args.turn_absolute:
        args.turn_heading_mode = "absolute"
    if args.list_skills:
        for skill in list_static_sonic_skills():
            lock = "requires_lock" if skill.requires_upper_body_lock else "no_lock"
            print(f"{skill.name}\t{lock}\t{skill.description}")
        raise SystemExit(0)
    return args


def _runtime_event_payload(args: argparse.Namespace, *, started_at: float) -> dict[str, object]:
    if args.skill == "carry_recover_payload_inward":
        return {
            "upper_body_event": "recover_payload_inward",
            "upper_body_event_id": args.upper_body_event_id
            or f"recover_payload_inward_{int(started_at * 1000)}",
            "upper_body_pose_name": PAYLOAD_RECOVERY_POSE_NAME,
            "upper_body_ramp_s": max(0.1, float(args.recovery_ramp_seconds)),
        }
    if args.skill == "release_payload":
        return {
            "payload_event": "release_payload",
            "payload_event_id": args.payload_event_id
            or f"release_payload_{int(started_at * 1000)}",
            "payload_release_ramp_s": max(0.1, float(args.release_ramp_seconds)),
        }
    if args.skill != "place_payload_on_arms":
        return {}
    payload: dict[str, object] = {
        "payload_event": "place_payload_on_arms",
        "payload_event_id": args.payload_event_id or f"place_payload_on_arms_{int(started_at * 1000)}",
        "payload_force": bool(args.payload_force),
    }
    for key, value in (
        ("payload_x", args.payload_x),
        ("payload_y", args.payload_y),
        ("payload_z", args.payload_z),
        ("payload_yaw", args.payload_yaw),
        ("payload_rearward_offset_m", args.payload_rearward_offset_m),
    ):
        if value is not None:
            payload[key] = float(value)
    return payload


def _resolve_turn_state_file(args: argparse.Namespace) -> Path | None:
    if args.skill not in {"turn_in_place", "carry_turn_in_place"}:
        return None
    if args.turn_heading_mode != "absolute_state":
        return None
    if args.turn_state_file is not None:
        return args.turn_state_file
    if args.heartbeat_file is not None:
        return args.heartbeat_file.parent / "turn_heading_state.json"
    return args.output_dir.parent / "turn_heading_state.json"


def _prepare_turn_heading_stream(
    stream,
    *,
    skill_name: str,
    state_file: Path | None,
    reset_state: bool,
    heading_mode: str,
):
    if skill_name not in {"turn_in_place", "carry_turn_in_place"}:
        return stream, None
    if heading_mode == "incremental":
        by_robot = stream.by_robot()
        first_robot_commands = next(iter(by_robot.values()), [])
        total = sum(
            float(command.delta_yaw_rad)
            for command in first_robot_commands
            if command.motion_axis == "yaw" and command.delta_yaw_rad is not None
        )
        return stream, {"mode": "incremental", "total_yaw_rad": total}
    if heading_mode == "absolute":
        return stream, {"mode": "absolute", "base_yaw_rad": 0.0, "final_yaw_rad": _stream_final_yaw(stream)}

    base_yaw = 0.0
    if state_file is not None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        if reset_state:
            state_file.unlink(missing_ok=True)
        elif state_file.exists():
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
                base_yaw = _wrap_angle(float(payload.get("cumulative_yaw_rad", 0.0)))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                base_yaw = 0.0

    adjusted_commands = []
    for command in stream.commands:
        if command.motion_axis != "yaw":
            adjusted_commands.append(command)
            continue
        if command.target_yaw_rad is None:
            adjusted_commands.append(command)
            continue
        target_yaw = _wrap_angle(base_yaw + float(command.target_yaw_rad))
        delta_yaw = None if command.delta_yaw_rad is None else target_yaw
        adjusted_commands.append(
            replace(
                command,
                facing=(cos(target_yaw), sin(target_yaw), 0.0),
                delta_yaw_rad=delta_yaw,
                target_yaw_rad=target_yaw,
            )
        )
    adjusted_stream = type(stream)(adjusted_commands)
    final_yaw = _stream_final_yaw(adjusted_stream)
    final_yaw = _wrap_angle(final_yaw)
    state = {
        "mode": "bounded_relative_state",
        "base_yaw_rad": base_yaw,
        "final_yaw_rad": final_yaw,
        "range": "[-pi, pi]",
    }
    if state_file is not None:
        state_file.write_text(
            json.dumps(
                {
                    "cumulative_yaw_rad": final_yaw,
                    "heading_state_is_wrapped": True,
                    "updated_unix": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return adjusted_stream, state


def _stream_final_yaw(stream) -> float:
    values = [command.target_yaw_rad for command in stream.commands if command.target_yaw_rad is not None]
    if not values:
        return 0.0
    return float(values[-1])


def _wrap_angle(angle: float) -> float:
    wrapped = (float(angle) + pi) % (2.0 * pi) - pi
    if wrapped <= -pi + 1.0e-12:
        return pi
    return wrapped


class _HeartbeatWriter:
    def __init__(self, path: Path | None, min_interval_s: float = 0.25) -> None:
        self.path = path
        self.min_interval_s = min_interval_s
        self._last_write = 0.0
        self._count = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def tick(self, phase: str, payload: dict[str, object] | None = None, *, force: bool = False) -> None:
        if self.path is None:
            return
        now = time.time()
        if not force and now - self._last_write < self.min_interval_s:
            return
        self._count += 1
        heartbeat = {"phase": phase, "count": self._count, "time_unix": now}
        if payload:
            heartbeat.update(payload)
        self.path.write_text(
            json.dumps(heartbeat),
            encoding="utf-8",
        )
        self._last_write = now


def _resolve_ports(args: argparse.Namespace, robot_ids: tuple[str, ...]) -> dict[str, int]:
    requested = {
        "front": args.front_port,
        "back": args.back_port,
        "g1": args.g1_port,
    }
    ports = {robot_id: int(requested[robot_id]) for robot_id in robot_ids}
    if len(set(ports.values())) != len(ports):
        raise ValueError(f"each active robot needs a distinct ZMQ port, got {ports}")
    return ports


def _publish_start_pulses(
    publishers: dict[str, SonicZmqPublisher],
    *,
    duration_s: float,
    publish_hz: float,
    heartbeat: _HeartbeatWriter,
    idle_commands: dict[str, SonicCommand] | None = None,
) -> int:
    if duration_s <= 0.0:
        return 0
    period = 1.0 / max(publish_hz, 1.0e-6)
    deadline = time.monotonic() + duration_s
    count = 0
    while True:
        for publisher in publishers.values():
            publisher.send_start_planner()
        if idle_commands is not None:
            send_synchronized_planner_commands(publishers, idle_commands)
        if idle_commands is None:
            heartbeat.tick("start_pulses")
        else:
            reference = next(iter(idle_commands.values()))
            heartbeat.tick(
                reference.phase_id,
                {
                    "skill_node": reference.skill_node,
                    "mode": int(reference.mode),
                    "motion_axis": reference.motion_axis,
                    "height": reference.height,
                    "duration_s": reference.duration_s,
                    "sync_group": reference.sync_group,
                    "startup_prime": True,
                },
            )
        count += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(period, remaining))
    return count


def _publish_stream_groups(
    publishers: dict[str, SonicZmqPublisher],
    commands: list[SonicCommand],
    *,
    publish_hz: float,
    heartbeat: _HeartbeatWriter,
    turn_heading_mode: str,
    runtime_event_payload: dict[str, object],
    safety_monitor: _PayloadSafetyMonitor | None = None,
    team_sync_controller: FormationSyncController | None = None,
    planar_hold: StationaryPlanarHold | None = None,
) -> list[dict[str, object]]:
    groups: list[list[SonicCommand]] = []
    for command in commands:
        if not groups or abs(groups[-1][0].t - command.t) > 1.0e-9:
            groups.append([command])
        else:
            groups[-1].append(command)
    counts = []
    captured_hold = False
    for group in groups:
        commands_by_robot = {command.robot_id: command for command in group}
        phase = group[0].phase_id or group[0].phase
        duration_s = max(command.duration_s for command in group)
        print(f"[sonic-static-skill] phase={phase} duration={duration_s:.1f}s", flush=True)
        if (
            planar_hold is not None
            and not captured_hold
            and group[0].motion_axis == "height"
        ):
            if planar_hold.capture():
                captured_hold = True
                print(
                    "[sonic-static-skill] captured place_crouch standing XY "
                    f"{planar_hold.last_diagnostics.get('reference_xy')}",
                    flush=True,
                )
            else:
                print(
                    "[sonic-static-skill] place_crouch XY capture skipped: "
                    f"{planar_hold.last_diagnostics.get('error')}",
                    flush=True,
                )
        counts.append(
            {
                "phase": phase,
                "messages_per_robot": _publish_command_group(
                    publishers,
                    commands_by_robot,
                    duration_s=duration_s,
                    publish_hz=publish_hz,
                    heartbeat=heartbeat,
                    turn_heading_mode=turn_heading_mode,
                    runtime_event_payload=runtime_event_payload,
                    safety_monitor=safety_monitor,
                    team_sync_controller=team_sync_controller,
                    planar_hold=planar_hold if captured_hold else None,
                ),
            }
        )
    return counts


def _publish_command_group(
    publishers: dict[str, SonicZmqPublisher],
    commands_by_robot: dict[str, SonicCommand],
    *,
    duration_s: float,
    publish_hz: float,
    heartbeat: _HeartbeatWriter,
    turn_heading_mode: str,
    runtime_event_payload: dict[str, object],
    safety_monitor: _PayloadSafetyMonitor | None = None,
    team_sync_controller: FormationSyncController | None = None,
    planar_hold: StationaryPlanarHold | None = None,
    completion_check: Callable[[], bool] | None = None,
) -> int:
    period = 1.0 / max(publish_hz, 1.0e-6)
    phase_started = time.monotonic()
    phase_duration = max(0.0, duration_s)
    deadline = phase_started + phase_duration
    group_reference = _heartbeat_reference_command(commands_by_robot)
    group_phase = group_reference.phase_id or group_reference.phase
    count = 0
    while True:
        if safety_monitor is not None:
            failure = safety_monitor.check()
            if failure:
                _publish_safety_abort(
                    publishers,
                    commands_by_robot,
                    heartbeat=heartbeat,
                    publish_hz=publish_hz,
                    reason=failure,
                )
                raise RuntimeError(f"physical safety abort during {group_phase}: {failure}")

        if count > 0 and completion_check is not None and completion_check():
            break

        elapsed_s = max(0.0, time.monotonic() - phase_started)
        phase_fraction = min(1.0, elapsed_s / max(phase_duration, 1.0e-9))
        adjusted_commands = _interpolate_profiled_speeds(
            commands_by_robot,
            phase_fraction=phase_fraction,
        )
        if team_sync_controller is not None:
            adjusted_commands, sync_failure = team_sync_controller.adjust(adjusted_commands)
            if sync_failure:
                _publish_safety_abort(
                    publishers,
                    commands_by_robot,
                    heartbeat=heartbeat,
                    publish_hz=publish_hz,
                    reason=sync_failure,
                )
                raise RuntimeError(
                    f"physical safety abort during {group_phase}: {sync_failure}"
                )
        if planar_hold is not None:
            adjusted_commands = planar_hold.adjust(adjusted_commands)

        publish_skew_s = send_synchronized_planner_commands(publishers, adjusted_commands)
        if team_sync_controller is not None:
            team_sync_controller.observe_publish_skew(publish_skew_s)
        for robot_id, publisher in publishers.items():
            command = adjusted_commands[robot_id]
            if command.delta_yaw_rad is not None or command.target_yaw_rad is not None:
                if turn_heading_mode == "incremental" and command.delta_yaw_rad is not None:
                    if count == 0:
                        publisher.send_heading_increment(command.delta_yaw_rad, planner=True)
                elif command.target_yaw_rad is not None and count == 0:
                    publisher.send_delta_heading(command.target_yaw_rad, planner=True)
        reference = _heartbeat_reference_command(adjusted_commands)
        heartbeat.tick(
            reference.phase_id,
            {
                "skill_node": reference.skill_node,
                "mode": int(reference.mode),
                "motion_axis": reference.motion_axis,
                "height": reference.height,
                "duration_s": reference.duration_s,
                "transport_ramp_s": reference.ramp_s,
                "commanded_speed_mps": reference.speed,
                "commanded_speed_mps_by_robot": {
                    robot_id: command.speed
                    for robot_id, command in adjusted_commands.items()
                },
                "host_publish_skew_ms": 1000.0 * publish_skew_s,
                "sync_group": reference.sync_group,
                **(
                    {"place_crouch_planar_hold": planar_hold.last_diagnostics}
                    if planar_hold is not None
                    else {}
                ),
                **(
                    runtime_event_payload
                    if reference.motion_axis in {"payload_event", "upper_body_event"}
                    or "payload" in (reference.phase_id or reference.phase)
                    else {}
                ),
            },
            force=bool(
                count == 0
                and (
                    reference.motion_axis in {"payload_event", "upper_body_event"}
                    or "payload" in (reference.phase_id or reference.phase)
                )
            ),
        )
        count += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(period, remaining))
    return count


def _interpolate_profiled_speeds(
    commands_by_robot: dict[str, SonicCommand],
    *,
    phase_fraction: float,
) -> dict[str, SonicCommand]:
    """Resolve optional speed profiles into planner-stable plateaus.

    SONIC replans whenever the positive speed override changes, while zero means
    "use the mode default" rather than stop. A few held plateaus therefore give
    the policy time to settle and avoid a high-frequency replan loop.
    """

    alpha = min(1.0, max(0.0, float(phase_fraction)))
    plateau_index = min(
        _PROFILE_SPEED_PLATEAUS - 1,
        int(alpha * _PROFILE_SPEED_PLATEAUS),
    )
    profile_alpha = plateau_index / max(1, _PROFILE_SPEED_PLATEAUS - 1)
    resolved: dict[str, SonicCommand] = {}
    for robot_id, command in commands_by_robot.items():
        if command.speed_start_mps is None and command.speed_end_mps is None:
            resolved[robot_id] = command
            continue
        start = (
            float(command.speed)
            if command.speed_start_mps is None
            else float(command.speed_start_mps)
        )
        end = start if command.speed_end_mps is None else float(command.speed_end_mps)
        resolved[robot_id] = replace(
            command,
            speed=start + (end - start) * profile_alpha,
        )
    return resolved


def _heartbeat_reference_command(
    commands_by_robot: dict[str, SonicCommand],
) -> SonicCommand:
    """Describe the task-changing command when an async peer is holding."""

    return max(
        commands_by_robot.values(),
        key=lambda command: (
            command.motion_axis == "payload_event",
            command.motion_axis in {"longitudinal", "lateral", "yaw", "height"},
            command.motion_axis != "hold",
        ),
    )


class _PayloadSafetyMonitor:
    """Monitor Term1 state while a carry command is physically executing."""

    def __init__(
        self,
        runtime_state_file: Path,
        *,
        max_age_s: float,
        min_payload_z_m: float,
        min_robot_pelvis_height_m: float,
        min_robot_xy_separation_m: float,
        max_tilt_deg: float,
        hard_tilt_deg: float,
        tilt_grace_s: float,
        contact_loss_grace_s: float,
        max_support_center_xy_error_m: float,
        min_support_force_per_robot_n: float,
        min_support_force_balance_ratio: float,
        force_imbalance_grace_s: float,
        max_top_surface_z_range_m: float = 0.20,
        wall_contact_is_hard_failure: bool = False,
        support_center_is_hard_failure: bool = False,
        strict_geometry_checks: bool = False,
        max_robot_xy_separation_m: float | None = None,
    ) -> None:
        self.runtime_state_file = Path(runtime_state_file)
        self.max_age_s = max(0.1, float(max_age_s))
        self.min_payload_z_m = float(min_payload_z_m)
        self.min_robot_pelvis_height_m = float(min_robot_pelvis_height_m)
        self.min_robot_xy_separation_m = max(
            0.0, float(min_robot_xy_separation_m)
        )
        self.max_tilt_deg = max(0.0, float(max_tilt_deg))
        self.hard_tilt_deg = max(self.max_tilt_deg, float(hard_tilt_deg))
        self.tilt_grace_s = max(0.0, float(tilt_grace_s))
        self.contact_loss_grace_s = max(0.0, float(contact_loss_grace_s))
        self.max_support_center_xy_error_m = max(
            0.0, float(max_support_center_xy_error_m)
        )
        self.min_support_force_per_robot_n = max(
            0.0, float(min_support_force_per_robot_n)
        )
        self.min_support_force_balance_ratio = min(
            1.0, max(0.0, float(min_support_force_balance_ratio))
        )
        self.force_imbalance_grace_s = max(0.0, float(force_imbalance_grace_s))
        self.max_top_surface_z_range_m = max(
            0.0, float(max_top_surface_z_range_m)
        )
        self.wall_contact_is_hard_failure = bool(wall_contact_is_hard_failure)
        self.support_center_is_hard_failure = bool(support_center_is_hard_failure)
        self.strict_geometry_checks = bool(strict_geometry_checks)
        self.max_robot_xy_separation_m = (
            None
            if max_robot_xy_separation_m is None
            else max(0.0, float(max_robot_xy_separation_m))
        )
        self.contact_loss_started: float | None = None
        self.force_imbalance_started: float | None = None
        self.tilt_exceeded_started: float | None = None
        self.min_seen_payload_z_m = float("inf")
        self.min_seen_robot_pelvis_height_m = float("inf")
        self.min_seen_robot_xy_separation_m = float("inf")
        self.max_seen_robot_xy_separation_m = 0.0
        self.max_seen_payload_tilt_deg = 0.0
        self.min_seen_support_force_n = float("inf")
        self.min_seen_support_force_balance_ratio = float("inf")
        self.max_seen_support_center_xy_error_m = 0.0
        self.max_contact_loss_s = 0.0
        self.max_force_imbalance_s = 0.0
        self.max_tilt_exceed_s = 0.0
        self.sample_count = 0
        self.last_status = PayloadRuntimeStatus(error="payload safety monitor has no sample")

    def observe(self, status: PayloadRuntimeStatus) -> None:
        self.last_status = status
        self.sample_count += 1
        self.min_seen_payload_z_m = min(self.min_seen_payload_z_m, status.z)
        self.min_seen_robot_pelvis_height_m = min(
            self.min_seen_robot_pelvis_height_m,
            status.min_robot_pelvis_height_m,
        )
        if status.robot_xy_separation_available:
            self.min_seen_robot_xy_separation_m = min(
                self.min_seen_robot_xy_separation_m,
                status.robot_xy_separation_m,
            )
            self.max_seen_robot_xy_separation_m = max(
                self.max_seen_robot_xy_separation_m,
                status.robot_xy_separation_m,
            )
        self.max_seen_payload_tilt_deg = max(
            self.max_seen_payload_tilt_deg,
            status.tilt_deg,
        )
        if status.support_center_xy_error_available:
            self.max_seen_support_center_xy_error_m = max(
                self.max_seen_support_center_xy_error_m,
                status.support_center_xy_error_m,
            )
        if status.support_force_available:
            self.min_seen_support_force_n = min(
                self.min_seen_support_force_n,
                *(status.support_normal_force_n_by_robot.get(robot_id, 0.0) for robot_id in ("front", "back")),
            )
            self.min_seen_support_force_balance_ratio = min(
                self.min_seen_support_force_balance_ratio,
                status.support_force_balance_ratio,
            )

    def check(self) -> str:
        status = read_payload_runtime_status(
            self.runtime_state_file,
            max_age_s=self.max_age_s,
        )
        self.observe(status)
        if not status.available or not status.fresh:
            return "runtime_state_missing_or_stale: " + status.describe()
        if not status.active:
            return "payload_not_active: " + status.describe()
        if status.min_robot_pelvis_height_m < self.min_robot_pelvis_height_m:
            return "robot_fall: " + status.describe()
        if status.floor_contact:
            return "payload_floor_contact: " + status.describe()
        if status.z <= self.min_payload_z_m:
            return "payload_below_safety_height: " + status.describe()
        if (
            self.strict_geometry_checks
            and status.tilt_deg >= PAYLOAD_TOP_FACE_UP_LIMIT_DEG
        ):
            return "payload_top_face_inverted: " + status.describe()
        if status.wall_contacts and self.wall_contact_is_hard_failure:
            return "payload_wall_contact: " + status.describe()
        if self.support_center_is_hard_failure:
            if not status.support_center_xy_error_available:
                return "payload_support_center_error_unavailable: " + status.describe()
            if status.support_center_xy_error_m > self.max_support_center_xy_error_m:
                return "payload_support_center_error_exceeded: " + status.describe()

        now = time.monotonic()
        if self.strict_geometry_checks:
            if status.top_surface_corner_z_range_m > self.max_top_surface_z_range_m:
                return "payload_tilt_exceeded: " + status.describe()
            if status.tilt_deg > self.hard_tilt_deg:
                return "payload_hard_tilt_exceeded: " + status.describe()
            if status.tilt_deg > self.max_tilt_deg:
                if self.tilt_exceeded_started is None:
                    self.tilt_exceeded_started = now
                elapsed = now - self.tilt_exceeded_started
                self.max_tilt_exceed_s = max(self.max_tilt_exceed_s, elapsed)
                if elapsed > self.tilt_grace_s:
                    return "payload_tilt_exceeded: " + status.describe()
            else:
                if self.tilt_exceeded_started is not None:
                    self.max_tilt_exceed_s = max(
                        self.max_tilt_exceed_s,
                        now - self.tilt_exceeded_started,
                    )
                self.tilt_exceeded_started = None

        both_supporting = {"front", "back"}.issubset(status.supporting_robots)
        if both_supporting:
            if self.contact_loss_started is not None:
                self.max_contact_loss_s = max(self.max_contact_loss_s, now - self.contact_loss_started)
            self.contact_loss_started = None
            if (
                self.strict_geometry_checks
                and status.support_force_available
                and not status.support_force_balanced(
                    min_force_per_robot_n=self.min_support_force_per_robot_n,
                    min_balance_ratio=self.min_support_force_balance_ratio,
                )
            ):
                if self.force_imbalance_started is None:
                    self.force_imbalance_started = now
                    return ""
                elapsed = now - self.force_imbalance_started
                self.max_force_imbalance_s = max(self.max_force_imbalance_s, elapsed)
                if elapsed > self.force_imbalance_grace_s:
                    return "payload_support_force_imbalanced: " + status.describe()
                return ""
            if self.force_imbalance_started is not None:
                self.max_force_imbalance_s = max(
                    self.max_force_imbalance_s,
                    now - self.force_imbalance_started,
                )
            self.force_imbalance_started = None
        else:
            self.force_imbalance_started = None
            if self.contact_loss_started is None:
                self.contact_loss_started = now
                return ""
            elapsed = now - self.contact_loss_started
            self.max_contact_loss_s = max(self.max_contact_loss_s, elapsed)
            if elapsed > self.contact_loss_grace_s:
                return "dual_payload_support_lost: " + status.describe()
            # Do not let a secondary spacing diagnostic mask the primary
            # physical event while the contact grace window is being evaluated.
            return ""

        if not status.robot_xy_separation_available:
            return "robot_pair_separation_unavailable: " + status.describe()
        if status.robot_xy_separation_m < self.min_robot_xy_separation_m:
            return "robot_pair_too_close: " + status.describe()
        if (
            self.max_robot_xy_separation_m is not None
            and status.robot_xy_separation_m > self.max_robot_xy_separation_m
        ):
            return "robot_pair_too_far: " + status.describe()
        return ""


    def report(self) -> dict[str, object]:
        report = {
            "sample_count": self.sample_count,
            "task_failure_contract": (
                "drop_or_dual_upper_limb_support_loss_or_robot_fall"
                if not self.strict_geometry_checks
                else "strict_geometry_and_support"
            ),
            "min_payload_z_m": 0.0 if self.min_seen_payload_z_m == float("inf") else self.min_seen_payload_z_m,
            "min_robot_pelvis_height_m": (
                0.0
                if self.min_seen_robot_pelvis_height_m == float("inf")
                else self.min_seen_robot_pelvis_height_m
            ),
            "min_robot_xy_separation_m": (
                0.0
                if self.min_seen_robot_xy_separation_m == float("inf")
                else self.min_seen_robot_xy_separation_m
            ),
            "robot_xy_separation_limit_m": self.min_robot_xy_separation_m,
            "max_robot_xy_separation_m": self.max_seen_robot_xy_separation_m,
            "robot_xy_max_separation_limit_m": self.max_robot_xy_separation_m,
            "max_payload_tilt_deg": self.max_seen_payload_tilt_deg,
            "max_support_center_xy_error_m": self.max_seen_support_center_xy_error_m,
            "max_contact_loss_s": self.max_contact_loss_s,
            "last_status": self.last_status.describe(),
        }
        if self.strict_geometry_checks:
            report.update(
                {
                    "payload_tilt_soft_limit_deg": self.max_tilt_deg,
                    "payload_tilt_hard_limit_deg": self.hard_tilt_deg,
                    "max_tilt_exceed_s": self.max_tilt_exceed_s,
                    "tilt_grace_s": self.tilt_grace_s,
                    "support_center_xy_error_limit_m": self.max_support_center_xy_error_m,
                    "min_support_force_per_robot_n": (
                        None
                        if self.min_seen_support_force_n == float("inf")
                        else self.min_seen_support_force_n
                    ),
                    "min_support_force_balance_ratio": (
                        None
                        if self.min_seen_support_force_balance_ratio == float("inf")
                        else self.min_seen_support_force_balance_ratio
                    ),
                    "support_force_per_robot_limit_n": self.min_support_force_per_robot_n,
                    "support_force_balance_ratio_limit": self.min_support_force_balance_ratio,
                    "max_force_imbalance_s": self.max_force_imbalance_s,
                    "force_imbalance_grace_s": self.force_imbalance_grace_s,
                }
            )
        return report


def _publish_safety_abort(
    publishers: dict[str, SonicZmqPublisher],
    commands_by_robot: dict[str, SonicCommand],
    *,
    heartbeat: _HeartbeatWriter,
    publish_hz: float,
    reason: str,
) -> None:
    period = 1.0 / max(float(publish_hz), 1.0e-6)
    idle_commands = {
        robot_id: _idle_command(
            robot_id=robot_id,
            facing=command.facing,
            height=command.height,
            duration_s=0.6,
            phase_id="physical_safety_abort",
            sync_group=command.sync_group,
        )
        for robot_id, command in commands_by_robot.items()
    }
    for _ in range(max(3, int(0.6 * float(publish_hz)))):
        send_synchronized_planner_commands(publishers, idle_commands)
        heartbeat.tick(
            "physical_safety_abort",
            {
                "skill_node": "physical_safety_abort",
                "mode": int(SonicLocomotionMode.IDLE),
                "motion_axis": "abort",
                "sync_group": next(iter(idle_commands.values())).sync_group,
                "failure_reason": reason,
            },
            force=True,
        )
        time.sleep(period)


def _wait_for_payload_readiness(
    publishers: dict[str, SonicZmqPublisher],
    idle_commands: dict[str, SonicCommand],
    *,
    heartbeat: _HeartbeatWriter,
    runtime_state_file: Path,
    timeout_s: float,
    stable_s: float,
    max_age_s: float,
    publish_hz: float,
) -> tuple[PayloadRuntimeStatus, int]:
    """Keep SONIC in IDLE while Term1's physical initialization gate settles."""

    period = 1.0 / max(float(publish_hz), 1.0e-6)
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stable_since: float | None = None
    last_log = 0.0
    count = 0
    status = PayloadRuntimeStatus(error=f"runtime state unavailable at {runtime_state_file}")
    while True:
        send_synchronized_planner_commands(publishers, idle_commands)
        reference = next(iter(idle_commands.values()))
        heartbeat.tick(
            reference.phase_id,
            {
                "skill_node": reference.skill_node,
                "mode": int(reference.mode),
                "motion_axis": reference.motion_axis,
                "height": reference.height,
                "sync_group": reference.sync_group,
                "payload_initialization_gate": "wait_robot_stable",
            },
        )
        count += 1

        now = time.monotonic()
        status = read_payload_runtime_status(runtime_state_file, max_age_s=max_age_s)
        ready = status.payload_supported(require_all_support_pads=True) or status.ready_to_place()
        if ready:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= max(0.0, float(stable_s)):
                return status, count
        else:
            stable_since = None
            if now - last_log >= 1.0:
                print(
                    "[sonic-static-skill] payload initialization gate waiting: "
                    + status.describe(),
                    flush=True,
                )
                last_log = now
        if now >= deadline:
            raise RuntimeError(
                "timed out waiting for stable carry-ready robots before payload placement: "
                + status.describe()
            )
        time.sleep(min(period, max(0.0, deadline - now)))


def _payload_status_json(
    status: PayloadRuntimeStatus,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "verified": status.payload_supported(
            min_z_m=args.payload_min_z,
            require_all_support_pads=args.payload_require_all_support_pads,
            safety_profile=args.payload_safety_profile,
        ),
        "z": status.z,
        "support_center_xy_error_available": status.support_center_xy_error_available,
        "support_center_xy_error_m": status.support_center_xy_error_m,
        "supporting_robots": sorted(status.supporting_robots),
        "supporting_geometries": sorted(status.supporting_geometries),
        "non_support_robot_contacts": list(status.non_support_robot_contacts),
        "support_polygon_available": status.support_polygon_available,
        "support_polygon_com_margin_m": status.support_polygon_com_margin_m,
        "floor_contact": status.floor_contact,
        "wall_contacts": list(status.wall_contacts),
        "runtime_state_age_s": status.age_s,
    }


def _physical_motion_report(
    stream,
    *,
    before: dict[str, object] | None,
    after: dict[str, object] | None,
    capture_errors: list[str],
) -> dict[str, object]:
    report: dict[str, object] = {
        "available": False,
        "capture_errors": list(capture_errors),
        "before": before,
        "after": after,
        "metrics": None,
    }
    if before is None or after is None:
        return report
    reference = next(
        (
            command
            for command in stream.sorted().commands
            if command.motion_axis in {"longitudinal", "lateral"}
            and (abs(command.movement[0]) > 1.0e-9 or abs(command.movement[1]) > 1.0e-9)
            and command.speed >= 0.0
        ),
        None,
    )
    if reference is None:
        report["capture_errors"] = [*capture_errors, "no translation command in stream"]
        return report
    try:
        report["metrics"] = compute_runtime_motion_metrics(
            before,
            after,
            target_direction_xy=(reference.movement[0], reference.movement[1]),
            reference_frame=reference.reference_frame,
        )
    except (RuntimeError, ValueError) as exc:
        report["capture_errors"] = [*capture_errors, str(exc)]
        return report
    report["available"] = True
    return report


def _idle_command(
    *,
    robot_id: str,
    facing: tuple[float, float, float],
    height: float,
    duration_s: float,
    phase_id: str,
    sync_group: str,
) -> SonicCommand:
    return SonicCommand(
        t=0.0,
        robot_id=robot_id,
        skill_node=phase_id,
        mode=SonicLocomotionMode.IDLE,
        movement=(0.0, 0.0, 0.0),
        facing=facing,
        speed=-1.0,
        height=height,
        duration_s=duration_s,
        phase=phase_id,
        phase_id=phase_id,
        sync_group=sync_group,
    )


if __name__ == "__main__":
    main()
