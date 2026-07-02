#!/usr/bin/env python3
"""Publish synchronized official GEAR-SONIC commands to two deploy backends."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_coop.sonic.commands import SonicCommand, SonicLocomotionMode  # noqa: E402
from praxis_coop.sonic.upper_body import (  # noqa: E402
    HAND_POSES,
    UPPER_BODY_JOINT_NAMES,
    UPPER_BODY_POSES,
    resolve_hand_pose,
    resolve_upper_body_pose,
)
from praxis_coop.sonic.zmq_publisher import SonicZmqConfig, SonicZmqPublisher  # noqa: E402


MODE_BY_NAME = {
    "idle": SonicLocomotionMode.IDLE,
    "slow_walk": SonicLocomotionMode.SLOW_WALK,
    "walk": SonicLocomotionMode.WALK,
    "run": SonicLocomotionMode.RUN,
    "object_carrying": SonicLocomotionMode.OBJECT_CARRYING,
}

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = _HeartbeatWriter(args.heartbeat_file)

    movement = _unit_xy(args.movement_x, args.movement_y)
    facing = _unit_xy(args.facing_x, args.facing_y)
    walk_seconds = args.walk_seconds
    if walk_seconds is None:
        if args.walk_speed <= 0.0:
            raise ValueError("--walk-speed must be positive when --walk-seconds is omitted")
        walk_seconds = args.walk_meters / args.walk_speed

    publishers = {
        "front": SonicZmqPublisher(
            SonicZmqConfig(
                bind_host=args.bind_host,
                port=args.front_port,
                startup_delay_s=args.startup_delay,
                publish_period_s=1.0 / args.publish_hz,
                realtime=True,
                verbose=args.verbose_zmq,
            )
        ),
        "back": SonicZmqPublisher(
            SonicZmqConfig(
                bind_host=args.bind_host,
                port=args.back_port,
                startup_delay_s=args.startup_delay,
                publish_period_s=1.0 / args.publish_hz,
                realtime=True,
                verbose=args.verbose_zmq,
            )
        ),
    }
    upper_body_pose_name = args.upper_body_pose
    hand_pose_name = args.hand_pose
    if args.hold_zero_upper_body:
        upper_body_pose_name = "zero"
    if args.hold_default_hands:
        hand_pose_name = "default"
    upper_body_position = resolve_upper_body_pose(upper_body_pose_name, args.upper_body_values)
    upper_body_start_position = resolve_upper_body_pose(args.upper_body_start_pose, None)
    left_hand_joints, right_hand_joints = resolve_hand_pose(
        hand_pose_name,
        args.left_hand_values,
        args.right_hand_values,
    )
    if upper_body_position is not None:
        print(
            f"[dual-sonic-preflight] upper_body_pose={upper_body_pose_name} "
            f"order={','.join(UPPER_BODY_JOINT_NAMES)} "
            f"values={_format_floats(upper_body_position)}",
            flush=True,
        )
    if left_hand_joints is not None and right_hand_joints is not None:
        print(
            f"[dual-sonic-preflight] hand_pose={hand_pose_name} "
            f"left={_format_floats(left_hand_joints)} right={_format_floats(right_hand_joints)}",
            flush=True,
        )
    counts: list[dict[str, object]] = []
    started_at = time.time()
    try:
        print(
            f"[dual-sonic-preflight] start pulses {args.start_pulse_seconds:.1f}s "
            f"front={args.front_port} back={args.back_port}",
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
                ),
            }
        )

        if args.stabilize_seconds > 0.0:
            stable_commands = {
                robot_id: _command(
                    robot_id=robot_id,
                    mode=SonicLocomotionMode.IDLE,
                    movement=(0.0, 0.0, 0.0),
                    facing=(facing[0], facing[1], 0.0),
                    speed=-1.0,
                    height=args.height,
                    duration_s=args.stabilize_seconds,
                    phase_id="dual_motor_preflight_stabilize_default",
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                    upper_body_position=upper_body_start_position,
                )
                for robot_id in publishers
            }
            print(
                f"[dual-sonic-preflight] stabilize with upper_body_start_pose="
                f"{args.upper_body_start_pose} for {args.stabilize_seconds:.1f}s",
                flush=True,
            )
            counts.append(
                {
                    "phase": "stabilize_default",
                    "messages_per_robot": _publish_planner_phase(
                        publishers,
                        stable_commands,
                        duration_s=args.stabilize_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                    ),
                }
            )

        if args.upper_body_ramp_seconds > 0.0 and upper_body_position is not None:
            if upper_body_start_position is None:
                raise ValueError("--upper-body-ramp-seconds requires a non-none --upper-body-start-pose")
            print(
                f"[dual-sonic-preflight] ramp upper body "
                f"{args.upper_body_start_pose}->{upper_body_pose_name} "
                f"over {args.upper_body_ramp_seconds:.1f}s",
                flush=True,
            )
            counts.append(
                {
                    "phase": "upper_body_ramp",
                    "messages_per_robot": _publish_upper_body_ramp_phase(
                        publishers,
                        mode=SonicLocomotionMode.IDLE,
                        facing=(facing[0], facing[1], 0.0),
                        speed=-1.0,
                        height=args.height,
                        start_upper_body=upper_body_start_position,
                        end_upper_body=upper_body_position,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                        duration_s=args.upper_body_ramp_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                    ),
                }
            )

        if args.stand_seconds > 0.0:
            idle_commands = {
                robot_id: _command(
                    robot_id=robot_id,
                    mode=SonicLocomotionMode.IDLE,
                    movement=(0.0, 0.0, 0.0),
                    facing=(facing[0], facing[1], 0.0),
                    speed=-1.0,
                    height=args.height,
                    duration_s=args.stand_seconds,
                    phase_id="dual_motor_preflight_stand",
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                    upper_body_position=upper_body_position,
                )
                for robot_id in publishers
            }
            print(
                f"[dual-sonic-preflight] carry pose hold ready; press 9 now if ElasticBand is still enabled. "
                f"holding for {args.stand_seconds:.1f}s",
                flush=True,
            )
            counts.append(
                {
                    "phase": "stand",
                    "messages_per_robot": _publish_planner_phase(
                        publishers,
                        idle_commands,
                        duration_s=args.stand_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                    ),
                }
            )

        if walk_seconds > 0.0 and args.walk_meters > 0.0:
            walk_commands = {
                robot_id: _command(
                    robot_id=robot_id,
                    mode=MODE_BY_NAME[args.walk_mode],
                    movement=(movement[0], movement[1], 0.0),
                    facing=(facing[0], facing[1], 0.0),
                    speed=args.walk_speed,
                    height=args.height,
                    duration_s=walk_seconds,
                    phase_id="dual_motor_preflight_sync_walk",
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                    upper_body_position=upper_body_position,
                )
                for robot_id in publishers
            }
            print(
                f"[dual-sonic-preflight] synchronized walk target={args.walk_meters:.2f}m "
                f"speed={args.walk_speed:.2f}m/s duration={walk_seconds:.1f}s",
                flush=True,
            )
            counts.append(
                {
                    "phase": "walk",
                    "messages_per_robot": _publish_planner_phase(
                        publishers,
                        walk_commands,
                        duration_s=walk_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                    ),
                }
            )

        if args.final_idle_seconds > 0.0:
            final_idle_commands = {
                robot_id: _command(
                    robot_id=robot_id,
                    mode=SonicLocomotionMode.IDLE,
                    movement=(0.0, 0.0, 0.0),
                    facing=(facing[0], facing[1], 0.0),
                    speed=-1.0,
                    height=args.height,
                    duration_s=args.final_idle_seconds,
                    phase_id="dual_motor_preflight_final_idle",
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                    upper_body_position=upper_body_position,
                )
                for robot_id in publishers
            }
            print(f"[dual-sonic-preflight] final idle {args.final_idle_seconds:.1f}s", flush=True)
            counts.append(
                {
                    "phase": "final_idle",
                    "messages_per_robot": _publish_planner_phase(
                        publishers,
                        final_idle_commands,
                        duration_s=args.final_idle_seconds,
                        publish_hz=args.publish_hz,
                        heartbeat=heartbeat,
                    ),
                }
            )

        if not args.no_stop:
            for publisher in publishers.values():
                publisher.send_stop()
    finally:
        for publisher in publishers.values():
            publisher.close()

    result = {
        "status": "dual_commands_sent_manual_visual_verification_required",
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "front_port": args.front_port,
        "back_port": args.back_port,
        "heartbeat_file": str(args.heartbeat_file) if args.heartbeat_file is not None else None,
        "upper_body_pose": upper_body_pose_name,
        "upper_body_start_pose": args.upper_body_start_pose,
        "upper_body_joint_order": list(UPPER_BODY_JOINT_NAMES),
        "upper_body_position": list(upper_body_position) if upper_body_position is not None else None,
        "hand_pose": hand_pose_name,
        "left_hand_joints": list(left_hand_joints) if left_hand_joints is not None else None,
        "right_hand_joints": list(right_hand_joints) if right_hand_joints is not None else None,
        "counts": counts,
        "success_criterion": [
            "two G1 robots remain upright after ElasticBand is disabled",
            "both robots execute the same walking phase at the same time",
            "upper-body and hand planner targets keep the payload-support pose stable",
            "no root-pose proxy, no Isaac fake success, no payload contact is involved in no-box tests",
        ],
    }
    output_file = args.output_dir / "dual_preflight_result.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[dual-sonic-preflight] wrote {output_file}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--front-port", type=int, default=5556)
    parser.add_argument("--back-port", type=int, default=5558)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--start-pulse-seconds", type=float, default=6.0)
    parser.add_argument("--start-pulse-hz", type=float, default=10.0)
    parser.add_argument("--stand-seconds", type=float, default=8.0)
    parser.add_argument("--walk-meters", type=float, default=1.0)
    parser.add_argument("--walk-seconds", type=float, default=None)
    parser.add_argument("--walk-speed", type=float, default=0.2)
    parser.add_argument("--walk-mode", choices=sorted(MODE_BY_NAME), default="slow_walk")
    parser.add_argument("--movement-x", type=float, default=1.0)
    parser.add_argument("--movement-y", type=float, default=0.0)
    parser.add_argument("--facing-x", type=float, default=1.0)
    parser.add_argument("--facing-y", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument(
        "--stabilize-seconds",
        type=float,
        default=0.0,
        help="Hold a stable upper-body start pose before ramping into the carrying pose.",
    )
    parser.add_argument(
        "--upper-body-start-pose",
        choices=sorted(("none", *UPPER_BODY_POSES)),
        default="sonic_default",
        help="Start pose used by --stabilize-seconds and --upper-body-ramp-seconds.",
    )
    parser.add_argument(
        "--upper-body-ramp-seconds",
        type=float,
        default=0.0,
        help="Smoothly interpolate from --upper-body-start-pose to --upper-body-pose before the main hold/walk.",
    )
    parser.add_argument("--final-idle-seconds", type=float, default=30.0)
    parser.add_argument(
        "--upper-body-pose",
        choices=sorted(("none", "custom", *UPPER_BODY_POSES)),
        default="none",
        help="17-DOF upper-body pose target sent with every planner command.",
    )
    parser.add_argument(
        "--upper-body-values",
        nargs=17,
        metavar="RAD",
        help="Custom 17-float upper-body target in the printed SONIC deploy order.",
    )
    parser.add_argument(
        "--hand-pose",
        choices=sorted(("none", "custom", *HAND_POSES)),
        default="none",
        help="Dex3 hand pose target sent with every planner command.",
    )
    parser.add_argument(
        "--left-hand-values",
        nargs=7,
        metavar="RAD",
        help="Custom left Dex3 7-float target; requires --hand-pose custom.",
    )
    parser.add_argument(
        "--right-hand-values",
        nargs=7,
        metavar="RAD",
        help="Custom right Dex3 7-float target; requires --hand-pose custom.",
    )
    parser.add_argument(
        "--hold-default-hands",
        action="store_true",
        help="Compatibility alias for --hand-pose default.",
    )
    parser.add_argument(
        "--hold-zero-upper-body",
        action="store_true",
        help="Compatibility alias for --upper-body-pose zero.",
    )
    parser.add_argument("--no-stop", action="store_true")
    parser.add_argument("--verbose-zmq", action="store_true")
    parser.add_argument(
        "--heartbeat-file",
        type=Path,
        default=None,
        help="Continuously touch/write this file while planner commands are being published.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "deployment" / "dual_official_sonic_motor_preflight",
    )
    return parser.parse_args()


class _HeartbeatWriter:
    def __init__(self, path: Path | None, min_interval_s: float = 0.25) -> None:
        self.path = path
        self.min_interval_s = min_interval_s
        self._last_write = 0.0
        self._count = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def tick(self, phase: str) -> None:
        if self.path is None:
            return
        now = time.time()
        if now - self._last_write < self.min_interval_s:
            return
        self._count += 1
        payload = {
            "phase": phase,
            "count": self._count,
            "time_unix": now,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        self._last_write = now


def _publish_start_pulses(
    publishers: dict[str, SonicZmqPublisher],
    *,
    duration_s: float,
    publish_hz: float,
    heartbeat: _HeartbeatWriter,
) -> int:
    if publish_hz <= 0.0:
        raise ValueError("--start-pulse-hz must be positive")
    period = 1.0 / publish_hz
    count = 0
    deadline = time.monotonic() + max(0.0, duration_s)
    while True:
        for publisher in publishers.values():
            publisher.send_start_planner()
        heartbeat.tick("start_pulses")
        count += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(period, remaining))
    return count


def _publish_planner_phase(
    publishers: dict[str, SonicZmqPublisher],
    commands: dict[str, SonicCommand],
    *,
    duration_s: float,
    publish_hz: float,
    heartbeat: _HeartbeatWriter,
) -> int:
    if publish_hz <= 0.0:
        raise ValueError("--publish-hz must be positive")
    period = 1.0 / publish_hz
    count = 0
    deadline = time.monotonic() + max(0.0, duration_s)
    while True:
        for robot_id, publisher in publishers.items():
            publisher.send_planner_command(commands[robot_id])
        heartbeat.tick(next(iter(commands.values())).phase_id)
        count += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(period, remaining))
    return count


def _publish_upper_body_ramp_phase(
    publishers: dict[str, SonicZmqPublisher],
    *,
    mode: SonicLocomotionMode,
    facing: tuple[float, float, float],
    speed: float,
    height: float,
    start_upper_body: tuple[float, ...],
    end_upper_body: tuple[float, ...],
    left_hand_joints: tuple[float, ...] | None,
    right_hand_joints: tuple[float, ...] | None,
    duration_s: float,
    publish_hz: float,
    heartbeat: _HeartbeatWriter,
) -> int:
    if publish_hz <= 0.0:
        raise ValueError("--publish-hz must be positive")
    if len(start_upper_body) != len(end_upper_body):
        raise ValueError("upper-body ramp endpoints must have the same length")
    period = 1.0 / publish_hz
    count = 0
    deadline = time.monotonic() + max(0.0, duration_s)
    while True:
        elapsed = max(0.0, duration_s - max(0.0, deadline - time.monotonic()))
        alpha = 1.0 if duration_s <= 1.0e-9 else min(1.0, elapsed / duration_s)
        alpha = _smoothstep(alpha)
        upper_body = tuple(
            (1.0 - alpha) * start + alpha * end for start, end in zip(start_upper_body, end_upper_body)
        )
        for robot_id, publisher in publishers.items():
            publisher.send_planner_command(
                _command(
                    robot_id=robot_id,
                    mode=mode,
                    movement=(0.0, 0.0, 0.0),
                    facing=facing,
                    speed=speed,
                    height=height,
                    duration_s=period,
                    phase_id="dual_motor_preflight_upper_body_ramp",
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                    upper_body_position=upper_body,
                )
            )
        heartbeat.tick("dual_motor_preflight_upper_body_ramp")
        count += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(period, remaining))
    return count


def _command(
    *,
    robot_id: str,
    mode: SonicLocomotionMode,
    movement: tuple[float, float, float],
    facing: tuple[float, float, float],
    speed: float,
    height: float,
    duration_s: float,
    phase_id: str,
    left_hand_joints: tuple[float, ...] | None = None,
    right_hand_joints: tuple[float, ...] | None = None,
    upper_body_position: tuple[float, ...] | None = None,
) -> SonicCommand:
    return SonicCommand(
        t=0.0,
        robot_id=robot_id,
        skill_node=phase_id,
        mode=mode,
        movement=movement,
        facing=facing,
        speed=speed,
        height=height,
        duration_s=duration_s,
        phase=phase_id,
        phase_id=phase_id,
        sync_group="dual_g1_motor_preflight",
        left_hand_joints=left_hand_joints,
        right_hand_joints=right_hand_joints,
        upper_body_position=upper_body_position,
        upper_body_velocity=(0.0,) * 17 if upper_body_position is not None else None,
    )


def _unit_xy(x: float, y: float) -> tuple[float, float]:
    norm = math.hypot(x, y)
    if norm <= 1e-9:
        raise ValueError("direction vector cannot be zero")
    return (x / norm, y / norm)


def _format_floats(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def _smoothstep(alpha: float) -> float:
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


if __name__ == "__main__":
    main()
