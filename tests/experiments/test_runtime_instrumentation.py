from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from coevolve_experiments.production.runtime_instrumentation import (
    MuJoCoTerminalTraceRecorder,
    RuntimeInstrumentationError,
    RuntimeTraceBinding,
    instrument_runtime_class,
    load_live_terminal_snapshot,
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _binding() -> RuntimeTraceBinding:
    return RuntimeTraceBinding(
        run_id="table3__mirrored__seed_200",
        seed=200,
        scene_fingerprint=_digest("scene"),
        scene_realization_sha256=_digest("realization"),
        compiled_model_sha256=_digest("model"),
        scene_execution_evidence_fingerprint=_digest("scene-evidence"),
        target_center_xyz_m=(-1.5, 7.6, 0.3),
        simulator_version="3.3.5",
        contact_role_registry_source_sha256=_digest("execution"),
        wall_prim_names=("topology_wall_01", "wall_01"),
        target_geom_names=("target_marker",),
        floor_geom_names=("floor", "manual_floor"),
    )


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


class _Model:
    def __init__(self) -> None:
        self._geoms = {
            1: _Named("long_box_geom"),
            2: _Named("target_marker"),
            3: _Named("manual_floor"),
            4: _Named("front_left_carry_support_plate"),
            5: _Named("mystery_fixture"),
            6: _Named("front_pelvis"),
            7: _Named("back_pelvis"),
        }
        self._bodies = {
            1: _Named("long_box"),
            2: _Named("target"),
            3: _Named("manual_floor"),
            4: _Named("front_left_hand"),
            5: _Named("mystery_fixture"),
            6: _Named("front_pelvis"),
            7: _Named("back_pelvis"),
        }
        self.geom_bodyid = [0, 1, 2, 3, 4, 5]

    def geom(self, index: int) -> _Named:
        return self._geoms[index]

    def body(self, index: int) -> _Named:
        return self._bodies[index]


class _Contact:
    def __init__(self, geom1: int, geom2: int, dist: float) -> None:
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist


class _Runtime:
    def __init__(self) -> None:
        self.model = _Model()
        self.payload_geom_id = 1
        self.target_geom_id = 2
        self.payload_qpos_adr = 0
        self.slots = (
            SimpleNamespace(
                spec=SimpleNamespace(robot_id="front"),
                root_qpos_adr=7,
                pelvis_body_id=6,
            ),
            SimpleNamespace(
                spec=SimpleNamespace(robot_id="back"),
                root_qpos_adr=14,
                pelvis_body_id=7,
            ),
        )
        self.data = SimpleNamespace(
            time=0.0,
            qpos=[
                -1.4,
                7.6,
                0.65,
                1.0,
                0.0,
                0.0,
                0.0,
                -0.2,
                7.6,
                0.75,
                1.0,
                0.0,
                0.0,
                0.0,
                -2.8,
                7.6,
                0.76,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            ncon=1,
            contact=[_Contact(1, 2, -0.001)],
            xpos=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-0.2, 7.6, 0.75],
                [-2.8, 7.6, 0.76],
            ],
            xquat=[[1.0, 0.0, 0.0, 0.0] for _ in range(8)],
        )
        self._checkpoint_status = {
            "checkpoint_id": "accepted-17",
            "event_id": "save:accepted-17:001",
        }
        self._payload_release_active = True
        self._post_release_crouched_retreat_completed = True
        self._post_release_stand_recovered = True


def test_binding_is_derived_from_consumed_channel_values() -> None:
    binding = RuntimeTraceBinding.from_consumed_channels(
        {
            "execution_composite_sha256": _digest("realization"),
            "execution_sha256": _digest("execution"),
            "compiled_model_sha256": _digest("model"),
            "execution_evidence_fingerprint": _digest("scene-evidence"),
            "contact_role_registry": {
                "schema_version": 1,
                "source_execution_sha256": _digest("execution"),
                "wall_prim_names": ["topology_wall_01", "wall_01"],
                "target_geom_names": ["target_marker"],
                "floor_geom_names": ["floor", "manual_floor"],
            },
            "target": {"center": [-1.5, 7.6, 0.3], "radius_m": 0.30},
        },
        {
            "run_spec": {
                "run_id": "table3__mirrored__seed_200",
                "seed": 200,
                "scene_fingerprint": _digest("scene"),
            }
        },
        simulator_version="3.3.5",
    )

    assert binding == _binding()
    assert binding.simulator["name"] == "MuJoCo"


def test_recorder_builds_full_pose_raw_contact_and_three_second_trace() -> None:
    runtime = _Runtime()
    recorder = MuJoCoTerminalTraceRecorder(_binding())
    for time_s in (0.0, 1.5, 3.0):
        runtime.data.time = time_s
        assert recorder.capture(runtime)

    documents = recorder.build_terminal_documents(require_stable_window=True)
    trace = documents.terminal_sample_trace
    state = documents.terminal_state

    assert trace["sample_window_s"] == 3.0
    assert trace["checkpoint_event_id"] == "save:accepted-17:001"
    assert trace["samples"][0]["payload_pose"]["quaternion_wxyz"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert trace["samples"][0]["contacts"] == [
        {
            "geom1_id": 1,
            "geom2_id": 2,
            "geom1_role": "payload",
            "geom2_role": "target",
            "distance_m": -0.001,
        }
    ]
    assert state["payload"] == {
        "released": True,
        "target_contact_all_samples": True,
        "floor_contact_samples": 0,
        "wall_contact_samples": 0,
        "robot_contact_samples": 0,
        "xy_error_m": pytest.approx(0.1),
        "z_error_m": 0.0,
        "tilt_deg": 0.0,
    }
    assert state["robots"]["front"]["upright"] is True
    assert state["robots"]["front"]["distance_to_target_m"] == 1.3


def test_checkpoint_change_discards_the_previous_window() -> None:
    runtime = _Runtime()
    recorder = MuJoCoTerminalTraceRecorder(_binding())
    runtime.data.time = 1.0
    recorder.capture(runtime)
    runtime._checkpoint_status["checkpoint_id"] = "accepted-18"
    runtime._checkpoint_status["event_id"] = "save:accepted-18:002"
    runtime.data.time = 2.0
    recorder.capture(runtime)

    assert recorder.checkpoint_id == "accepted-18"
    assert recorder.sample_count == 1
    with pytest.raises(RuntimeInstrumentationError, match="at least two"):
        recorder.build_terminal_documents()


def test_restore_event_change_discards_same_checkpoint_window() -> None:
    runtime = _Runtime()
    recorder = MuJoCoTerminalTraceRecorder(_binding())
    runtime.data.time = 1.0
    recorder.capture(runtime)
    runtime._checkpoint_status["event_id"] = "restore:accepted-17:002"
    runtime.data.time = 2.0
    recorder.capture(runtime)

    assert recorder.checkpoint_id == "accepted-17"
    assert recorder.checkpoint_event_id == "restore:accepted-17:002"
    assert recorder.sample_count == 1
    with pytest.raises(RuntimeInstrumentationError, match="at least two"):
        recorder.build_terminal_documents()


def test_hash_bound_topology_wall_role_is_recognized() -> None:
    runtime = _Runtime()
    runtime.model._geoms[5] = _Named("topology_wall_01_collision")
    runtime.model._bodies[5] = _Named("topology_wall_01")
    runtime.data.contact = [_Contact(1, 5, -0.001)]
    recorder = MuJoCoTerminalTraceRecorder(_binding())

    assert recorder.capture(runtime)
    runtime.data.time = 1.0
    assert recorder.capture(runtime)
    documents = recorder.build_terminal_documents()
    assert documents.terminal_state["payload"]["wall_contact_samples"] == 2


def test_unknown_payload_contact_role_fails_closed() -> None:
    runtime = _Runtime()
    runtime.data.contact = [_Contact(1, 5, -0.001)]
    recorder = MuJoCoTerminalTraceRecorder(_binding())

    with pytest.raises(RuntimeInstrumentationError, match="no reviewed role"):
        recorder.capture(runtime)


def test_mixin_observes_after_base_publication_and_writes_reloadable_snapshot(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Base(_Runtime):
        def __init__(self, *, marker: str) -> None:
            super().__init__()
            events.append(marker)

        def _write_runtime_state(self) -> None:
            events.append("base-write")

    recorder = MuJoCoTerminalTraceRecorder(_binding())
    instrumented = instrument_runtime_class(Base)
    snapshot = (tmp_path / "live_terminal_snapshot.json").resolve()
    runtime = instrumented(
        marker="base-init",
        terminal_trace_recorder=recorder,
        terminal_trace_snapshot_path=snapshot,
    )
    for time_s in (0.0, 1.5, 3.0):
        runtime.data.time = time_s
        runtime._write_runtime_state()

    assert events == ["base-init", "base-write", "base-write", "base-write"]
    loaded = load_live_terminal_snapshot(snapshot)
    assert loaded.terminal_state["sample_count"] == 3
    assert loaded.terminal_state["sample_window_s"] == 3.0


def test_trace_does_not_silently_normalize_a_bad_quaternion() -> None:
    runtime = _Runtime()
    runtime.data.qpos[3:7] = [2.0, 0.0, 0.0, 0.0]
    recorder = MuJoCoTerminalTraceRecorder(_binding())

    with pytest.raises(RuntimeInstrumentationError, match="not unit length"):
        recorder.capture(runtime)
