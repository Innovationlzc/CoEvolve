#!/usr/bin/env python3
"""Publish one atomically timed sync or async team skill phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.sonic.team_coordination import (  # noqa: E402
    CoordinatedTeamPlan,
    TRANSLATION_METHODS,
    build_coordinated_schedule,
    schedule_duration,
)
from praxis_coop.sonic.safety import (  # noqa: E402
    CARRY_SAFETY_PROFILES,
    get_carry_safety_profile,
    loaded_turn_support_guard,
)
from praxis_coop.sonic.alignment_feedback import (  # noqa: E402
    RelativeAlignmentFeedback,
)
from praxis_coop.sonic.linear_distance_feedback import (  # noqa: E402
    LINEAR_FEEDBACK_TIMEOUT_GAIN,
    LinearDistanceFeedback,
)
from praxis_coop.sonic.runtime_state import read_payload_runtime_status  # noqa: E402
from praxis_coop.sonic.team_sync import (  # noqa: E402
    FormationSyncConfig,
    FormationSyncController,
)
from praxis_coop.sonic.turn_feedback import PayloadYawFeedback  # noqa: E402
from praxis_coop.sonic.zmq_publisher import SonicZmqConfig, SonicZmqPublisher  # noqa: E402
from run_sonic_static_skill import (  # noqa: E402
    _HeartbeatWriter,
    _PayloadSafetyMonitor,
    _idle_command,
    _publish_command_group,
    _publish_start_pulses,
)


_PAYLOAD_TURN_METHODS = frozenset(
    {
        "coordinated_arc_transport",
        "coordinated_pivot",
    }
)
_PAYLOAD_TURN_AXES = frozenset({"arc", "pivot"})
_ALIGNMENT_TRANSLATION_AXES = frozenset({"longitudinal", "lateral"})
_ALIGNMENT_TRANSLATION_MAX_WINDOW_S = 4.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.plan_file.read_text(encoding="utf-8"))
    plan = CoordinatedTeamPlan.from_payload(payload)
    groups = build_coordinated_schedule(
        plan,
        minimum_transport_window_s=args.minimum_transport_window_s,
        minimum_lateral_transport_window_s=args.minimum_lateral_transport_window_s,
        transport_duration_gain=args.transport_duration_gain,
        transport_duration_baseline_m=args.transport_duration_baseline_m,
        minimum_transport_ramp_s=args.minimum_transport_ramp_s,
    )
    (args.output_dir / "compiled_coordinated_schedule.json").write_text(
        json.dumps(
            [
                {
                    "duration_s": max(command.duration_s for command in group.values()),
                    "commands": {
                        robot_id: command.to_json()
                        for robot_id, command in group.items()
                    },
                }
                for group in groups
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    publishers = {
        "front": SonicZmqPublisher(
            SonicZmqConfig(
                port=args.front_port,
                startup_delay_s=args.startup_delay,
                publish_period_s=1.0 / args.publish_hz,
                realtime=True,
                verbose=args.verbose_zmq,
            )
        ),
        "back": SonicZmqPublisher(
            SonicZmqConfig(
                port=args.back_port,
                startup_delay_s=args.startup_delay,
                publish_period_s=1.0 / args.publish_hz,
                realtime=True,
                verbose=args.verbose_zmq,
            )
        ),
    }
    heartbeat = _HeartbeatWriter(args.heartbeat_file)
    safety_monitor = None
    turn_guard: dict[str, object] = {"enabled": False}
    if args.runtime_state_file is not None:
        method = plan.actions["front"].method
        max_support_center_error_m = args.payload_safety_max_support_center_error_m
        support_center_is_hard_failure = args.payload_support_center_is_hard_failure
        max_robot_xy_separation_m = None
        if method in _PAYLOAD_TURN_METHODS:
            safety = get_carry_safety_profile(args.payload_safety_profile)
            (
                max_support_center_error_m,
                support_center_is_hard_failure,
            ) = loaded_turn_support_guard(
                safety,
                max_support_center_error_m,
            )
            status = read_payload_runtime_status(
                args.runtime_state_file,
                max_age_s=args.runtime_state_max_age_s,
            )
            if status.robot_xy_separation_available:
                max_robot_xy_separation_m = (
                    status.robot_xy_separation_m
                    + safety.warning_robot_step_desync_m
                )
            turn_guard = {
                "enabled": True,
                "method": method,
                "support_center_xy_error_limit_m": max_support_center_error_m,
                "support_center_is_hard_failure": support_center_is_hard_failure,
                "initial_robot_xy_separation_m": (
                    status.robot_xy_separation_m
                    if status.robot_xy_separation_available
                    else None
                ),
                "max_robot_xy_separation_m": max_robot_xy_separation_m,
            }
            print(
                "[sonic-coordination] loaded-turn guard enabled "
                f"support_offset<={max_support_center_error_m:.3f}m "
                f"support_offset_mode={'hard' if support_center_is_hard_failure else 'diagnostic'} "
                f"robot_separation<={max_robot_xy_separation_m}",
                flush=True,
            )
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
            max_support_center_xy_error_m=max_support_center_error_m,
            min_support_force_per_robot_n=args.payload_support_min_force_n,
            min_support_force_balance_ratio=args.payload_support_min_balance_ratio,
            force_imbalance_grace_s=args.payload_force_imbalance_grace_s,
            max_top_surface_z_range_m=args.payload_safety_max_top_surface_range_m,
            wall_contact_is_hard_failure=args.payload_wall_contact_is_hard_failure,
            support_center_is_hard_failure=support_center_is_hard_failure,
            strict_geometry_checks=args.payload_safety_profile == "strict",
            max_robot_xy_separation_m=max_robot_xy_separation_m,
        )
    team_sync_controller = None
    if (
        args.runtime_state_file is not None
        and not args.no_team_sync
        and plan.coordination_mode == "sync"
        and all(action.method in TRANSLATION_METHODS for action in plan.actions.values())
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
        print(
            "[sonic-coordination] live formation controller enabled "
            f"active_catchup={args.team_sync_active_catchup} "
            f"pause={args.team_sync_pause_leader_error_m:.3f}m "
            f"resume={args.team_sync_resume_error_m:.3f}m",
            flush=True,
        )

    started = time.time()
    phase_counts: list[dict[str, object]] = []
    turn_feedback = None
    alignment_feedback = None
    linear_distance_feedback = None
    linear_feedback_result: dict[str, object] = {"enabled": False}
    error = ""
    try:
        idle = {
            robot_id: _idle_command(
                robot_id=robot_id,
                facing=(1.0, 0.0, 0.0),
                height=0.80,
                duration_s=max(args.start_pulse_seconds, args.stand_seconds, 0.5),
                phase_id=f"{plan.phase_id}.pre_stand",
                sync_group=plan.synchronization_group,
            )
            for robot_id in ("front", "back")
        }
        _publish_start_pulses(
            publishers,
            duration_s=args.start_pulse_seconds,
            publish_hz=args.publish_hz,
            heartbeat=heartbeat,
            idle_commands=idle,
        )
        if args.stand_seconds > 0.0:
            count = _publish_command_group(
                publishers,
                idle,
                duration_s=args.stand_seconds,
                publish_hz=args.publish_hz,
                heartbeat=heartbeat,
                turn_heading_mode="incremental",
                runtime_event_payload={},
            )
            phase_counts.append({"phase": f"{plan.phase_id}.pre_stand", "messages": count})

        turn_method = plan.actions["front"].method
        if (
            args.runtime_state_file is not None
            and plan.coordination_mode == "sync"
            and turn_method in TRANSLATION_METHODS
            and all(
                action.method == turn_method
                for action in plan.actions.values()
            )
        ):
            parameters = plan.actions["front"].parameters
            semantic_direction = _semantic_translation_direction(
                turn_method,
                parameters,
            )
            target_distance_m = min(
                abs(float(action.parameters["distance_m"]))
                for action in plan.actions.values()
            )
            command_speed_mps = min(
                abs(float(action.parameters["speed_mps"]))
                for action in plan.actions.values()
            )
            motion_command = _first_translation_command(groups)
            linear_distance_feedback = LinearDistanceFeedback(
                args.runtime_state_file,
                target_distance_m=target_distance_m,
                target_direction_xy=(
                    motion_command.movement[0],
                    motion_command.movement[1],
                ),
                reference_frame=motion_command.reference_frame,
                primitive=turn_method,
                semantic_direction=semantic_direction,
                loaded_context="loaded",
                command_speed_mps=command_speed_mps,
                calibration_file=args.linear_response_calibration_file,
                max_state_age_s=args.runtime_state_max_age_s,
            )
            print(
                "[sonic-coordination] measured linear endpoint feedback enabled "
                f"target={target_distance_m:.3f}m "
                f"stop={linear_distance_feedback.stop_threshold_m:.3f}m "
                f"key={linear_distance_feedback.calibration_key}",
                flush=True,
            )
        if (
            args.runtime_state_file is not None
            and turn_method in _PAYLOAD_TURN_METHODS
        ):
            parameters = plan.actions["front"].parameters
            expected_direction = (
                -1 if parameters.get("turn_direction") == "clockwise" else 1
            )
            turn_feedback = PayloadYawFeedback(
                args.runtime_state_file,
                expected_direction=expected_direction,
                max_state_age_s=args.runtime_state_max_age_s,
            )
            print(
                "[sonic-coordination] payload-yaw feedback enabled "
                f"target={expected_direction * float(parameters['angle_deg']):.2f}deg "
                f"margin={turn_feedback.stop_margin_rad:.4f}rad",
                flush=True,
            )
        if (
            args.runtime_state_file is not None
            and turn_method == "coordinated_alignment"
        ):
            axes = {
                str(action.parameters.get("axis", ""))
                for action in plan.actions.values()
            }
            moving_robots = [
                robot_id
                for robot_id, action in plan.actions.items()
                if abs(float(action.parameters.get("delta", 0.0))) > 1.0e-9
            ]
            if len(axes) == 1 and next(iter(axes)) in _ALIGNMENT_TRANSLATION_AXES:
                if len(moving_robots) != 1:
                    raise RuntimeError(
                        "feedback-bounded translation alignment requires one moving "
                        "robot and one reference robot"
                    )
                mover_robot = moving_robots[0]
                reference_robot = (
                    "back" if mover_robot == "front" else "front"
                )
                mover_parameters = plan.actions[mover_robot].parameters
                alignment_feedback = RelativeAlignmentFeedback(
                    args.runtime_state_file,
                    mover_robot=mover_robot,
                    reference_robot=reference_robot,
                    axis=next(iter(axes)),
                    signed_target_delta_m=float(mover_parameters["delta"]),
                    max_state_age_s=args.runtime_state_max_age_s,
                )
                print(
                    "[sonic-coordination] relative alignment feedback enabled "
                    f"mover={mover_robot} reference={reference_robot} "
                    f"axis={alignment_feedback.axis} "
                    f"target={alignment_feedback.signed_target_delta_m:.3f}m "
                    f"max_window={_ALIGNMENT_TRANSLATION_MAX_WINDOW_S:.1f}s",
                    flush=True,
                )

        print(
            f"[sonic-coordination] mode={plan.coordination_mode} phase={plan.phase_id} "
            f"duration={schedule_duration(groups):.2f}s",
            flush=True,
        )
        for group_index, group in enumerate(groups, start=1):
            duration = max(command.duration_s for command in group.values())
            alignment_translation_active = bool(
                alignment_feedback is not None
                and any(
                    command.skill_node == "coordinated_alignment"
                    and command.motion_axis in _ALIGNMENT_TRANSLATION_AXES
                    and command.speed > 0.0
                    for command in group.values()
                )
            )
            if alignment_translation_active:
                duration = max(duration, _ALIGNMENT_TRANSLATION_MAX_WINDOW_S)
            linear_translation_active = bool(
                linear_distance_feedback is not None
                and any(
                    command.motion_axis in {"longitudinal", "lateral"}
                    and command.speed > 0.0
                    for command in group.values()
                )
            )
            if linear_translation_active:
                duration = max(
                    duration,
                    LINEAR_FEEDBACK_TIMEOUT_GAIN
                    * linear_distance_feedback.target_distance_m
                    / max(linear_distance_feedback.command_speed_mps, 1.0e-6),
                )
            methods = {
                robot_id: command.skill_node
                for robot_id, command in group.items()
            }
            print(
                f"[sonic-coordination] interval={group_index:02d} "
                f"duration={duration:.2f}s actions={methods}",
                flush=True,
            )
            turn_targets = {
                float(command.target_yaw_rad)
                for command in group.values()
                if command.motion_axis in _PAYLOAD_TURN_AXES
                and command.target_yaw_rad is not None
            }
            if len(turn_targets) > 1:
                raise RuntimeError(
                    f"turn interval {group_index} has inconsistent payload yaw targets"
                )
            turn_target = next(iter(turn_targets), None)
            if (
                turn_feedback is not None
                and turn_target is not None
                and turn_feedback.reached(turn_target)
            ):
                phase_counts.append(
                    {
                        "phase": f"{plan.phase_id}.interval_{group_index:02d}",
                        "messages": 0,
                        "actions": methods,
                        "payload_yaw_target_rad": turn_target,
                        "completed_before_publish": True,
                    }
                )
                continue
            count = _publish_command_group(
                publishers,
                group,
                duration_s=duration,
                publish_hz=args.publish_hz,
                heartbeat=heartbeat,
                turn_heading_mode="incremental",
                runtime_event_payload={},
                safety_monitor=safety_monitor,
                team_sync_controller=team_sync_controller,
                completion_check=(
                    (
                        lambda: alignment_feedback.reached()
                    )
                    if alignment_translation_active
                    else (
                        (lambda: linear_distance_feedback.reached())
                        if linear_translation_active
                        else (
                            None
                            if turn_feedback is None or turn_target is None
                            else lambda target=turn_target: turn_feedback.reached(target)
                        )
                    )
                ),
            )
            if (
                linear_distance_feedback is not None
                and linear_distance_feedback.trigger_payload_progress_m is not None
                and team_sync_controller is not None
            ):
                team_sync_controller.lock_translation_endpoint()
            turn_progress = (
                turn_feedback.progress_rad()
                if turn_feedback is not None and turn_target is not None
                else None
            )
            phase_counts.append(
                {
                    "phase": f"{plan.phase_id}.interval_{group_index:02d}",
                    "messages": count,
                    "actions": methods,
                    "payload_yaw_target_rad": turn_target,
                    "payload_yaw_progress_rad": turn_progress,
                    "relative_alignment_progress_m": (
                        alignment_feedback.latest_progress_m
                        if alignment_translation_active
                        else None
                    ),
                }
            )
        if args.post_transport_idle_s > 0.0:
            final_idle = {
                robot_id: _idle_command(
                    robot_id=robot_id,
                    facing=(1.0, 0.0, 0.0),
                    height=0.80,
                    duration_s=args.post_transport_idle_s,
                    phase_id=f"{plan.phase_id}.post_idle",
                    sync_group=plan.synchronization_group,
                )
                for robot_id in ("front", "back")
            }
            count = _publish_command_group(
                publishers,
                final_idle,
                duration_s=args.post_transport_idle_s,
                publish_hz=args.publish_hz,
                heartbeat=heartbeat,
                turn_heading_mode="incremental",
                runtime_event_payload={},
                safety_monitor=safety_monitor,
                team_sync_controller=team_sync_controller,
            )
            settled_turn_progress = (
                turn_feedback.progress_rad()
                if turn_feedback is not None
                else None
            )
            phase_counts.append(
                {
                    "phase": f"{plan.phase_id}.post_idle",
                    "messages": count,
                    "settled_payload_yaw_progress_rad": settled_turn_progress,
                }
            )
        if linear_distance_feedback is not None:
            linear_feedback_result = linear_distance_feedback.finalize(
                physical_transaction_completed=not bool(error),
            )
    except Exception as exc:  # noqa: BLE001 - persist the physical abort reason
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if (
            linear_distance_feedback is not None
            and not linear_feedback_result.get("enabled", False)
        ):
            linear_feedback_result = linear_distance_feedback.finalize(
                physical_transaction_completed=not bool(error),
            )
        for publisher in publishers.values():
            publisher.close()
        result = {
            "status": "failed" if error else "success",
            "error": error,
            "coordination_mode": plan.coordination_mode,
            "phase_id": plan.phase_id,
            "synchronization_group": plan.synchronization_group,
            "actions": {
                robot_id: {
                    "method": action.method,
                    "parameters": action.parameters,
                }
                for robot_id, action in plan.actions.items()
            },
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "phase_counts": phase_counts,
            "payload_safety_monitor": (
                safety_monitor.report() if safety_monitor is not None else {"enabled": False}
            ),
            "team_synchronization": (
                team_sync_controller.report()
                if team_sync_controller is not None
                else {"enabled": False}
            ),
            "payload_yaw_feedback": (
                turn_feedback.report()
                if turn_feedback is not None
                else {"enabled": False}
            ),
            "relative_alignment_feedback": (
                alignment_feedback.report()
                if alignment_feedback is not None
                else {"enabled": False}
            ),
            "linear_distance_feedback": linear_feedback_result,
            "loaded_turn_guard": turn_guard,
        }
        (args.output_dir / "coordinated_team_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heartbeat-file", type=Path, required=True)
    parser.add_argument("--runtime-state-file", type=Path, default=None)
    parser.add_argument(
        "--linear-response-calibration-file",
        type=Path,
        default=None,
    )
    parser.add_argument("--front-port", type=int, default=5556)
    parser.add_argument("--back-port", type=int, default=5558)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--startup-delay", type=float, default=0.5)
    parser.add_argument("--start-pulse-seconds", type=float, default=0.0)
    parser.add_argument("--stand-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-transport-window-s", type=float, default=1.0)
    parser.add_argument("--minimum-lateral-transport-window-s", type=float, default=0.4)
    parser.add_argument("--transport-duration-gain", type=float, default=1.0)
    parser.add_argument("--transport-duration-baseline-m", type=float, default=2.0)
    parser.add_argument("--minimum-transport-ramp-s", type=float, default=1.5)
    parser.add_argument("--post-transport-idle-s", type=float, default=1.0)
    parser.add_argument("--runtime-state-max-age-s", type=float, default=1.0)
    parser.add_argument(
        "--payload-safety-profile",
        choices=tuple(CARRY_SAFETY_PROFILES),
        default="task",
    )
    parser.add_argument("--payload-safety-min-z", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-safety-min-pelvis-height", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--payload-safety-min-robot-separation",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--payload-safety-max-tilt-deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-safety-hard-tilt-deg", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-tilt-grace-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-contact-loss-grace-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--payload-safety-max-support-center-error-m",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--payload-support-min-force-n", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-support-min-balance-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--payload-force-imbalance-grace-s", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-team-sync", action="store_true")
    parser.add_argument("--team-sync-active-catchup", action="store_true")
    parser.add_argument("--team-sync-longitudinal-kp", type=float, default=2.0)
    parser.add_argument("--team-sync-cross-track-kp", type=float, default=0.6)
    parser.add_argument("--team-sync-payload-kp", type=float, default=1.2)
    parser.add_argument("--team-sync-error-filter-alpha", type=float, default=0.35)
    parser.add_argument("--team-sync-pause-leader-error-m", type=float, default=0.008)
    parser.add_argument("--team-sync-resume-error-m", type=float, default=0.003)
    parser.add_argument("--team-sync-lagging-boost-mps", type=float, default=0.03)
    parser.add_argument("--team-sync-max-correction-mps", type=float, default=0.06)
    parser.add_argument("--team-sync-min-active-speed-ratio", type=float, default=0.0)
    parser.add_argument("--team-sync-hard-desync-m", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verbose-zmq", action="store_true")
    args = parser.parse_args()
    safety = get_carry_safety_profile(args.payload_safety_profile)
    args.payload_safety_min_z = (
        safety.min_payload_z_m if args.payload_safety_min_z is None else args.payload_safety_min_z
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
    if args.publish_hz <= 0.0:
        parser.error("--publish-hz must be positive")
    if args.front_port == args.back_port:
        parser.error("front and back require distinct ZMQ ports")
    if args.team_sync_resume_error_m < 0.0:
        parser.error("--team-sync-resume-error-m must be non-negative")
    if args.team_sync_pause_leader_error_m <= args.team_sync_resume_error_m:
        parser.error("--team-sync-pause-leader-error-m must exceed resume error")
    if (
        args.team_sync_hard_desync_m is not None
        and args.team_sync_hard_desync_m <= args.team_sync_pause_leader_error_m
    ):
        parser.error("--team-sync-hard-desync-m must exceed pause error")
    if not 0.0 <= args.team_sync_min_active_speed_ratio <= 1.0:
        parser.error("--team-sync-min-active-speed-ratio must be within [0, 1]")
    return args


def _first_translation_command(groups):
    for group in groups:
        for command in group.values():
            if (
                command.motion_axis in {"longitudinal", "lateral"}
                and command.speed > 0.0
                and (abs(command.movement[0]) > 1.0e-9 or abs(command.movement[1]) > 1.0e-9)
            ):
                return command
    raise RuntimeError("linear endpoint feedback found no translation command")


def _semantic_translation_direction(method: str, parameters: dict[str, object]) -> str:
    if method == "coordinated_linear_transport":
        return str(parameters["direction"]).strip().lower()
    mapping = {
        "carry_forward": "forward",
        "carry_backward": "backward",
        "carry_lateral_left": "left",
        "carry_lateral_right": "right",
    }
    try:
        return mapping[method]
    except KeyError as exc:
        raise ValueError(f"unsupported linear feedback method {method!r}") from exc


if __name__ == "__main__":
    main()
