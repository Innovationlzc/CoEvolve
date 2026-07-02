"""ZMQ publisher for GEAR-SONIC's zmq_manager input type."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from typing import Sequence

from praxis_coop.sonic.commands import SonicCommand, SonicCommandStream


HEADER_SIZE = 1280


@dataclass(frozen=True)
class SonicZmqConfig:
    """Connection settings for a SONIC deploy `zmq_manager` subscriber."""

    bind_host: str = "*"
    port: int = 5556
    startup_delay_s: float = 0.5
    publish_period_s: float = 0.1
    realtime: bool = True
    verbose: bool = True


class SonicZmqPublisher:
    """Publish command and planner messages using SONIC's packed ZMQ format."""

    def __init__(self, config: SonicZmqConfig | None = None) -> None:
        self.config = config or SonicZmqConfig()
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "pyzmq is required for --send-sonic. Install it with:\n"
                "/home/admin1/Project/IsaacLab/env_isaaclab/bin/python -m pip install pyzmq"
            ) from exc

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self.endpoint = f"tcp://{self.config.bind_host}:{self.config.port}"
        self._socket.bind(self.endpoint)
        time.sleep(self.config.startup_delay_s)
        if self.config.verbose:
            print(f"[sonic-zmq] bound {self.endpoint}")

    def close(self) -> None:
        self._socket.close()
        self._context.term()

    def send_start_planner(self) -> None:
        self._socket.send(build_command_message(start=True, stop=False, planner=True))
        if self.config.verbose:
            print("[sonic-zmq] command start=True planner=True")

    def send_stop(self) -> None:
        self._socket.send(build_command_message(start=False, stop=True, planner=True))
        if self.config.verbose:
            print("[sonic-zmq] command stop=True")

    def send_planner_command(self, command: SonicCommand) -> None:
        self._socket.send(
            build_planner_message(
                mode=int(command.mode),
                movement=command.movement,
                facing=command.facing,
                speed=command.speed,
                height=command.height,
                left_hand_joints=command.left_hand_joints,
                right_hand_joints=command.right_hand_joints,
                upper_body_position=command.upper_body_position,
                upper_body_velocity=command.upper_body_velocity,
                vr_position=command.vr_position,
                vr_orientation=command.vr_orientation,
            )
        )
        if self.config.verbose:
            print(
                "[sonic-zmq] "
                f"t={command.t:.2f} robot={command.robot_id} "
                f"mode={int(command.mode)} phase={command.phase_id or command.phase} "
                f"move=({command.movement[0]:.2f},{command.movement[1]:.2f}) "
                f"speed={command.speed:.2f} duration={command.duration_s:.2f}s"
            )

    def send_planner_command_for_duration(
        self,
        command: SonicCommand,
        *,
        duration_s: float | None = None,
        publish_hz: float | None = None,
    ) -> int:
        """Keep publishing a planner command so SONIC's planner input stays live."""

        duration = command.duration_s if duration_s is None else float(duration_s)
        if duration <= 0.0:
            self.send_planner_command(command)
            return 1

        period = self.config.publish_period_s
        if publish_hz is not None:
            if publish_hz <= 0.0:
                raise ValueError("publish_hz must be positive")
            period = 1.0 / float(publish_hz)
        period = max(0.001, float(period))

        count = 0
        deadline = time.monotonic() + duration
        while True:
            self.send_planner_command(command)
            count += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if self.config.realtime:
                time.sleep(min(period, remaining))
            else:
                break
        return count

    def publish_stream(self, stream: SonicCommandStream) -> None:
        """Publish a command stream.

        A single SONIC deploy process controls one robot. For two physical deploy
        instances, filter `stream.by_robot()[robot_id]` and run one publisher per
        robot/port. This method publishes the merged stream for smoke testing or
        a single active subscriber.
        """

        self.send_start_planner()
        commands = stream.sorted().commands
        if not commands:
            return
        start_wall = time.monotonic()
        for command in commands:
            if self.config.realtime:
                target_wall = start_wall + command.t
                delay = target_wall - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self.send_planner_command(command)
            if self.config.realtime:
                time.sleep(min(command.duration_s, self.config.publish_period_s))
        self.send_stop()


def build_command_message(
    *,
    start: bool,
    stop: bool,
    planner: bool,
    delta_heading: float | None = None,
) -> bytes:
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = (
        struct.pack("B", 1 if start else 0)
        + struct.pack("B", 1 if stop else 0)
        + struct.pack("B", 1 if planner else 0)
    )
    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))
    return b"command" + _build_header(fields) + payload


def build_planner_message(
    *,
    mode: int,
    movement: Sequence[float],
    facing: Sequence[float],
    speed: float = -1.0,
    height: float = -1.0,
    left_hand_joints: Sequence[float] | None = None,
    right_hand_joints: Sequence[float] | None = None,
    upper_body_position: Sequence[float] | None = None,
    upper_body_velocity: Sequence[float] | None = None,
    vr_position: Sequence[float] | None = None,
    vr_orientation: Sequence[float] | None = None,
) -> bytes:
    if len(movement) != 3:
        raise ValueError("movement must have length 3")
    if len(facing) != 3:
        raise ValueError("facing must have length 3")
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = (
        struct.pack("<i", int(mode))
        + struct.pack("<fff", float(movement[0]), float(movement[1]), float(movement[2]))
        + struct.pack("<fff", float(facing[0]), float(facing[1]), float(facing[2]))
        + struct.pack("<f", float(speed))
        + struct.pack("<f", float(height))
    )
    for name, values, length in (
        ("upper_body_position", upper_body_position, 17),
        ("upper_body_velocity", upper_body_velocity, 17),
        ("left_hand_joints", left_hand_joints, 7),
        ("right_hand_joints", right_hand_joints, 7),
        ("vr_position", vr_position, 9),
        ("vr_orientation", vr_orientation, 12),
    ):
        if values is None:
            continue
        payload += _pack_float_array_field(fields, name, values, length)
    return b"planner" + _build_header(fields) + payload


def _pack_float_array_field(
    fields: list[dict[str, object]],
    name: str,
    values: Sequence[float],
    length: int,
) -> bytes:
    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {len(values)}")
    fields.append({"name": name, "dtype": "f32", "shape": [length]})
    return struct.pack("<" + "f" * length, *(float(value) for value in values))


def _build_header(fields: list[dict[str, object]]) -> bytes:
    header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) > HEADER_SIZE:
        raise ValueError(f"ZMQ header too large: {len(encoded)} > {HEADER_SIZE}")
    return encoded.ljust(HEADER_SIZE, b"\x00")
