"""Pauseable dual-SONIC IDLE publisher for slow planning and VLM phases."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time
from typing import Callable, Iterator, Protocol

from .commands import SonicCommand, SonicLocomotionMode
from .zmq_publisher import SonicZmqConfig, SonicZmqPublisher


class _PlannerPublisher(Protocol):
    def send_planner_command(self, command: SonicCommand) -> None: ...

    def close(self) -> None: ...


PublisherFactory = Callable[[SonicZmqConfig], _PlannerPublisher]


@dataclass
class DualSonicIdleKeepalive:
    """Publish planner IDLE while the high-level agent is thinking.

    SONIC's two ZMQ manager ports each allow one bound publisher. Callers must
    stop this keepalive before dispatching a skill and restart it immediately
    afterwards. The class never sends START, STOP, or planner-reset messages.
    """

    heartbeat_file: Path
    front_port: int = 5556
    back_port: int = 5558
    publish_hz: float = 20.0
    phase_id: str = "cap_agent_idle_keepalive"
    startup_delay_s: float = 0.3
    publisher_factory: PublisherFactory = SonicZmqPublisher
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _ready_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)
    _heartbeat_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _heartbeat_suspend_count: int = field(default=0, init=False, repr=False)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and self._error is None

    def start(self, *, timeout_s: float = 3.0) -> None:
        if self.running:
            return
        if self.publish_hz <= 0.0:
            raise ValueError("idle keepalive publish_hz must be positive")
        self.stop()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="dual-sonic-idle-keepalive",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(max(0.1, float(timeout_s))):
            self.stop()
            raise RuntimeError("dual SONIC idle keepalive did not bind and publish in time")
        if self._error is not None:
            error = self._error
            self.stop()
            raise RuntimeError(f"dual SONIC idle keepalive failed: {error}") from error

    def stop(self, *, timeout_s: float = 3.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(max(0.1, float(timeout_s)))
        if thread.is_alive():
            raise RuntimeError("dual SONIC idle keepalive did not release its ZMQ ports")
        self._thread = None

    close = stop

    @contextmanager
    def suspend_heartbeat_writes(self) -> Iterator[None]:
        """Keep publishing IDLE while leasing the heartbeat file to a caller."""

        with self._heartbeat_lock:
            self._heartbeat_suspend_count += 1
        try:
            yield
        finally:
            with self._heartbeat_lock:
                self._heartbeat_suspend_count = max(
                    0,
                    self._heartbeat_suspend_count - 1,
                )

    def _run(self) -> None:
        publishers: dict[str, _PlannerPublisher] = {}
        try:
            period_s = 1.0 / float(self.publish_hz)
            publishers = {
                "front": self.publisher_factory(
                    SonicZmqConfig(
                        port=self.front_port,
                        startup_delay_s=self.startup_delay_s,
                        publish_period_s=period_s,
                        realtime=True,
                        verbose=False,
                    )
                ),
                "back": self.publisher_factory(
                    SonicZmqConfig(
                        port=self.back_port,
                        startup_delay_s=self.startup_delay_s,
                        publish_period_s=period_s,
                        realtime=True,
                        verbose=False,
                    )
                ),
            }
            commands = {
                robot_id: SonicCommand(
                    t=0.0,
                    robot_id=robot_id,
                    skill_node=self.phase_id,
                    mode=SonicLocomotionMode.IDLE,
                    movement=(0.0, 0.0, 0.0),
                    facing=(1.0, 0.0, 0.0),
                    speed=-1.0,
                    height=-1.0,
                    duration_s=period_s,
                    phase=self.phase_id,
                    phase_id=self.phase_id,
                    sync_group="dual_g1_agent_idle",
                )
                for robot_id in publishers
            }
            while not self._stop_event.is_set():
                for robot_id in ("front", "back"):
                    publishers[robot_id].send_planner_command(commands[robot_id])
                self._count += 1
                with self._heartbeat_lock:
                    if self._heartbeat_suspend_count == 0:
                        self._write_heartbeat()
                self._ready_event.set()
                self._stop_event.wait(period_s)
        except BaseException as exc:  # noqa: BLE001 - relay thread failures to the owner
            self._error = exc
            self._ready_event.set()
        finally:
            for publisher in publishers.values():
                try:
                    publisher.close()
                except Exception:
                    pass

    def _write_heartbeat(self) -> None:
        path = Path(self.heartbeat_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": self.phase_id,
            "count": self._count,
            "time_unix": time.time(),
        }
        temporary = path.with_name(f".{path.name}.keepalive.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)


__all__ = ["DualSonicIdleKeepalive"]
