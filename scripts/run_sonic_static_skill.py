#!/usr/bin/env python3
"""Publish one static SONIC skill to one or two official deploy backends."""

from __future__ import annotations

import argparse
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
from praxis_coop.sonic.zmq_publisher import SonicZmqConfig, SonicZmqPublisher  # noqa: E402


DEFAULT_PORTS = {
    "g1": 5556,
    "front": 5556,
    "back": 5558,
}


def main() -> None:
    args = parse_args()
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
            place_rise=args.place_rise,
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
                "--upper-body-lock-pose elbow110_carry_ready; this publisher does not send planner-side arm targets",
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
                    ),
                }
            )

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
            )
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
        "status": "static_sonic_skill_commands_sent_manual_visual_verification_required",
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
        "success_criterion": [
            "official MuJoCo runtime remains upright after ElasticBand drop",
            "the selected robot(s) execute the requested skill under official SONIC control",
            "for carry_* skills, Term1 motor-side upper-body lock keeps arms/hands unchanged",
            "no root-pose proxy, no Isaac fake success, and no payload result is claimed here",
        ],
    }
    output_file = args.output_dir / f"{args.skill}_result.json"
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[sonic-static-skill] wrote {output_file}", flush=True)


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
    parser.add_argument("--place-rise", action="store_true")
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
    if args.turn_absolute:
        args.turn_heading_mode = "absolute"
    if args.list_skills:
        for skill in list_static_sonic_skills():
            lock = "requires_lock" if skill.requires_upper_body_lock else "no_lock"
            print(f"{skill.name}\t{lock}\t{skill.description}")
        raise SystemExit(0)
    return args


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

    def tick(self, phase: str, payload: dict[str, object] | None = None) -> None:
        if self.path is None:
            return
        now = time.time()
        if now - self._last_write < self.min_interval_s:
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
    period = 1.0 / max(publish_hz, 1.0e-6)
    deadline = time.monotonic() + max(0.0, duration_s)
    count = 0
    while True:
        for robot_id, publisher in publishers.items():
            publisher.send_start_planner()
            if idle_commands is not None:
                publisher.send_planner_command(idle_commands[robot_id])
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
) -> list[dict[str, object]]:
    groups: list[list[SonicCommand]] = []
    for command in commands:
        if not groups or abs(groups[-1][0].t - command.t) > 1.0e-9:
            groups.append([command])
        else:
            groups[-1].append(command)
    counts = []
    for group in groups:
        commands_by_robot = {command.robot_id: command for command in group}
        phase = group[0].phase_id or group[0].phase
        duration_s = max(command.duration_s for command in group)
        print(f"[sonic-static-skill] phase={phase} duration={duration_s:.1f}s", flush=True)
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
) -> int:
    period = 1.0 / max(publish_hz, 1.0e-6)
    deadline = time.monotonic() + max(0.0, duration_s)
    count = 0
    while True:
        for robot_id, publisher in publishers.items():
            command = commands_by_robot[robot_id]
            publisher.send_planner_command(command)
            if command.motion_axis == "yaw":
                if turn_heading_mode == "incremental" and command.delta_yaw_rad is not None:
                    if count == 0:
                        publisher.send_heading_increment(command.delta_yaw_rad, planner=True)
                elif command.target_yaw_rad is not None and count == 0:
                    publisher.send_delta_heading(command.target_yaw_rad, planner=True)
        reference = next(iter(commands_by_robot.values()))
        heartbeat.tick(
            reference.phase_id,
            {
                "skill_node": reference.skill_node,
                "mode": int(reference.mode),
                "motion_axis": reference.motion_axis,
                "height": reference.height,
                "duration_s": reference.duration_s,
                "sync_group": reference.sync_group,
            },
        )
        count += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(period, remaining))
    return count


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
