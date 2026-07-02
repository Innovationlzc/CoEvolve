#!/usr/bin/env python3
"""Publish a minimal official GEAR-SONIC motor preflight command stream.

This script talks to ``gear_sonic_deploy --input-type zmq_manager``.  It does
not touch Qwen, SkillGraph, Isaac root-pose proxies, or the Python joint bridge.
The success criterion is visual/physical: the official MuJoCo SONIC robot must
stand for the requested duration and then walk straight without falling.
"""

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
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    movement = _unit_xy(args.movement_x, args.movement_y)
    facing = _unit_xy(args.facing_x, args.facing_y)
    walk_duration_s = args.walk_seconds
    if walk_duration_s is None:
        if args.walk_speed <= 0.0:
            raise ValueError("--walk-speed must be positive when --walk-seconds is omitted")
        walk_duration_s = args.walk_meters / args.walk_speed

    plan = {
        "backend": "official_gear_sonic_deploy_zmq_manager",
        "port": args.port,
        "start_pulse_seconds": args.start_pulse_seconds,
        "stand_seconds": args.stand_seconds,
        "walk_meters": args.walk_meters,
        "walk_seconds": walk_duration_s,
        "walk_speed": args.walk_speed,
        "walk_mode": args.walk_mode,
        "publish_hz": args.publish_hz,
        "movement": [movement[0], movement[1], 0.0],
        "facing": [facing[0], facing[1], 0.0],
        "success_criterion": [
            "official MuJoCo window shows the G1 standing without collapse for the stand phase",
            "official MuJoCo window shows the G1 walking approximately straight during the walk phase",
            "no proxy/root teleport/Isaac Python joint bridge result is counted as success",
        ],
    }
    (output_dir / "preflight_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    publisher = SonicZmqPublisher(
        SonicZmqConfig(
            bind_host=args.bind_host,
            port=args.port,
            startup_delay_s=args.startup_delay,
            publish_period_s=1.0 / args.publish_hz,
            realtime=True,
            verbose=args.verbose_zmq,
        )
    )
    sent_counts: list[dict[str, object]] = []
    started_at = time.time()

    try:
        print(
            f"[official-sonic-preflight] sending start pulses for {args.start_pulse_seconds:.1f}s "
            "in planner mode",
            flush=True,
        )
        start_count = _send_start_pulses(
            publisher,
            duration_s=args.start_pulse_seconds,
            publish_hz=args.start_pulse_hz,
        )
        sent_counts.append({"phase": "start_pulses", "messages": start_count})

        if args.stand_seconds > 0.0:
            print(
                f"[official-sonic-preflight] stand/idle for {args.stand_seconds:.1f}s "
                f"at {args.publish_hz:.1f} Hz",
                flush=True,
            )
            idle = SonicCommand(
                t=0.0,
                robot_id="g1",
                skill_node="official_preflight_stand_10s",
                mode=SonicLocomotionMode.IDLE,
                movement=(0.0, 0.0, 0.0),
                facing=(facing[0], facing[1], 0.0),
                speed=-1.0,
                height=args.height,
                duration_s=args.stand_seconds,
                phase="motor_preflight_stand",
                phase_id="motor_preflight_stand",
            )
            count = publisher.send_planner_command_for_duration(
                idle,
                duration_s=args.stand_seconds,
                publish_hz=args.publish_hz,
            )
            sent_counts.append({"phase": "stand", "messages": count})

        if walk_duration_s > 0.0 and args.walk_meters > 0.0:
            print(
                f"[official-sonic-preflight] walk {args.walk_meters:.2f}m target "
                f"with speed={args.walk_speed:.2f}m/s for {walk_duration_s:.1f}s",
                flush=True,
            )
            walk = SonicCommand(
                t=args.stand_seconds,
                robot_id="g1",
                skill_node="official_preflight_walk_1m",
                mode=MODE_BY_NAME[args.walk_mode],
                movement=(movement[0], movement[1], 0.0),
                facing=(facing[0], facing[1], 0.0),
                speed=args.walk_speed,
                height=args.height,
                duration_s=walk_duration_s,
                phase="motor_preflight_walk",
                phase_id="motor_preflight_walk",
                motion_axis="longitudinal",
            )
            count = publisher.send_planner_command_for_duration(
                walk,
                duration_s=walk_duration_s,
                publish_hz=args.publish_hz,
            )
            sent_counts.append({"phase": "walk", "messages": count})

        if args.final_idle_seconds > 0.0:
            print(
                f"[official-sonic-preflight] final idle for {args.final_idle_seconds:.1f}s",
                flush=True,
            )
            final_idle = SonicCommand(
                t=args.stand_seconds + walk_duration_s,
                robot_id="g1",
                skill_node="official_preflight_final_idle",
                mode=SonicLocomotionMode.IDLE,
                movement=(0.0, 0.0, 0.0),
                facing=(facing[0], facing[1], 0.0),
                speed=-1.0,
                height=args.height,
                duration_s=args.final_idle_seconds,
                phase="motor_preflight_final_idle",
                phase_id="motor_preflight_final_idle",
            )
            count = publisher.send_planner_command_for_duration(
                final_idle,
                duration_s=args.final_idle_seconds,
                publish_hz=args.publish_hz,
            )
            sent_counts.append({"phase": "final_idle", "messages": count})

        if not args.no_stop:
            print("[official-sonic-preflight] sending stop command", flush=True)
            publisher.send_stop()
    finally:
        publisher.close()

    result = {
        "status": "commands_sent_manual_visual_verification_required",
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
        "sent_counts": sent_counts,
        "plan_file": str(output_dir / "preflight_plan.json"),
    }
    (output_dir / "preflight_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[official-sonic-preflight] wrote {output_dir / 'preflight_result.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--startup-delay", type=float, default=1.0)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument("--start-pulse-seconds", type=float, default=6.0)
    parser.add_argument("--start-pulse-hz", type=float, default=10.0)
    parser.add_argument("--stand-seconds", type=float, default=10.0)
    parser.add_argument("--walk-meters", type=float, default=1.0)
    parser.add_argument("--walk-seconds", type=float, default=None)
    parser.add_argument("--walk-speed", type=float, default=0.2)
    parser.add_argument("--walk-mode", choices=sorted(MODE_BY_NAME), default="slow_walk")
    parser.add_argument("--movement-x", type=float, default=1.0)
    parser.add_argument("--movement-y", type=float, default=0.0)
    parser.add_argument("--facing-x", type=float, default=1.0)
    parser.add_argument("--facing-y", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=-1.0)
    parser.add_argument("--final-idle-seconds", type=float, default=2.0)
    parser.add_argument("--no-stop", action="store_true")
    parser.add_argument("--verbose-zmq", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "deployment" / "official_sonic_motor_preflight",
    )
    return parser.parse_args()


def _unit_xy(x: float, y: float) -> tuple[float, float]:
    norm = math.hypot(x, y)
    if norm <= 1e-9:
        raise ValueError("direction vector cannot be zero")
    return (x / norm, y / norm)


def _send_start_pulses(
    publisher: SonicZmqPublisher,
    *,
    duration_s: float,
    publish_hz: float,
) -> int:
    if duration_s <= 0.0:
        publisher.send_start_planner()
        return 1
    if publish_hz <= 0.0:
        raise ValueError("--start-pulse-hz must be positive")

    count = 0
    period = 1.0 / publish_hz
    deadline = time.monotonic() + duration_s
    while True:
        publisher.send_start_planner()
        count += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(period, remaining))
    return count


if __name__ == "__main__":
    main()
