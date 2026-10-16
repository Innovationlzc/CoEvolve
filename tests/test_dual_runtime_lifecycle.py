"""Small unit tests for native Viewer/runtime lifecycle helpers.

The production module imports SONIC's DDS bindings, which intentionally live in
Term1's separate environment. These tests extract the real method definitions
so the repository test environment still exercises their control flow.
"""

import ast
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


class _FakeMujoco:
    class FatalError(Exception):
        pass

    def __init__(self):
        self.contact_force = np.zeros(6, dtype=np.float64)
        self.step_error = None

    def mj_contactForce(self, _model, _data, _index, destination):
        destination[:] = self.contact_force

    def mj_step(self, _model, _data):
        if self.step_error is not None:
            raise self.step_error


FAKE_MUJOCO = _FakeMujoco()


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/praxis_coop/sonic_mujoco/dual_runtime.py"
)
METHOD_NAMES = {
    "_retire_closed_interactive_viewer",
    "_close_interactive_viewer",
    "_wait_for_viewer_native_teardown",
    "_runtime_numerical_fault",
    "_latch_numerical_fault",
    "_advance_physics_step",
    "_write_runtime_state",
}


def _load_lifecycle_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DualG1SonicMujocoRuntime"
    )
    methods = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in METHOD_NAMES
    ]
    namespace = {
        "Any": Any,
        "Path": Path,
        "json": json,
        "time": time,
        "np": np,
        "mujoco": FAKE_MUJOCO,
        "MAX_RUNTIME_CONTACT_FORCE_N": 1.0e6,
        "MAX_RUNTIME_ABS_QVEL": 1.0e4,
        "MAX_RUNTIME_ABS_QACC": 1.0e6,
    }
    module = ast.Module(
        body=[ast.ClassDef(name="RuntimeLifecycle", bases=[], keywords=[], body=methods, decorator_list=[])],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["RuntimeLifecycle"]


DualG1SonicMujocoRuntime = _load_lifecycle_class()


class _FakeViewer:
    def __init__(self, *, running: bool):
        self.running = running
        self.close_calls = 0

    def is_running(self):
        return self.running

    def close(self):
        self.close_calls += 1
        self.running = False


def test_user_closed_viewer_detaches_without_stopping_runtime(monkeypatch):
    runtime = object.__new__(DualG1SonicMujocoRuntime)
    viewer = _FakeViewer(running=False)
    runtime.viewer = viewer
    runtime._running = True
    runtime._interactive_viewer_closed = False
    monkeypatch.setattr(runtime, "_wait_for_viewer_native_teardown", lambda _viewer: None)

    runtime._retire_closed_interactive_viewer()

    assert runtime.viewer is None
    assert runtime._running is True
    assert runtime._interactive_viewer_closed is True
    assert viewer.close_calls == 0


def test_runtime_close_requests_live_viewer_shutdown_only_once(monkeypatch):
    runtime = object.__new__(DualG1SonicMujocoRuntime)
    viewer = _FakeViewer(running=True)
    runtime.viewer = viewer
    monkeypatch.setattr(runtime, "_wait_for_viewer_native_teardown", lambda _viewer: None)

    runtime._close_interactive_viewer()
    runtime._close_interactive_viewer()

    assert runtime.viewer is None
    assert viewer.close_calls == 1


def test_runtime_state_sequence_identifies_one_live_term1_instance(tmp_path):
    runtime = object.__new__(DualG1SonicMujocoRuntime)
    runtime.runtime_state_file = tmp_path / "runtime_state.json"
    runtime._runtime_state_sequence = 9
    runtime._runtime_instance_id = "term1-test"
    runtime._runtime_state_payload = lambda: {
        "runtime_instance_id": runtime._runtime_instance_id,
        "state_sequence": runtime._runtime_state_sequence,
    }

    runtime._write_runtime_state()

    assert runtime._runtime_state_sequence == 10
    assert runtime.runtime_state_file.read_text(encoding="utf-8") == (
        '{"runtime_instance_id": "term1-test", "state_sequence": 10}'
    )


def test_extreme_native_contact_force_latches_numerical_freeze():
    runtime = object.__new__(DualG1SonicMujocoRuntime)
    runtime.model = object()
    runtime.data = SimpleNamespace(
        time=12.5,
        ncon=1,
        qpos=np.zeros(2),
        qvel=np.zeros(2),
        qacc=np.zeros(2),
        ctrl=np.zeros(2),
        actuator_force=np.zeros(2),
        qfrc_constraint=np.zeros(2),
    )
    runtime._numerical_fault_latched = False
    runtime._runtime_fault = {}
    FAKE_MUJOCO.contact_force[:] = [1.9e17, 0.0, 0.0, 0.0, 0.0, 0.0]

    diagnostics = runtime._runtime_numerical_fault(phase="post_step")
    runtime._latch_numerical_fault(diagnostics)

    assert diagnostics["phase"] == "post_step"
    assert "contact_force>1000000N" in diagnostics["reasons"]
    assert runtime._numerical_fault_latched is True
    assert runtime._runtime_fault["type"] == "numerical_instability"
    FAKE_MUJOCO.contact_force[:] = 0.0


def test_native_mujoco_fatal_is_latched_without_stopping_runtime(monkeypatch):
    runtime = object.__new__(DualG1SonicMujocoRuntime)
    runtime.model = object()
    runtime.data = SimpleNamespace(
        time=12.5,
        ncon=0,
        qpos=np.zeros(2),
        qvel=np.zeros(2),
        qacc=np.zeros(2),
        ctrl=np.zeros(2),
        actuator_force=np.zeros(2),
        qfrc_constraint=np.zeros(2),
    )
    runtime._running = True
    runtime._numerical_fault_latched = False
    runtime._runtime_fault = {}
    writes = []
    monkeypatch.setattr(
        runtime,
        "_maybe_write_runtime_state_while_paused",
        lambda: writes.append(True),
        raising=False,
    )
    FAKE_MUJOCO.step_error = FAKE_MUJOCO.FatalError("rank-deficient Hessian")

    try:
        advanced = runtime._advance_physics_step()
    finally:
        FAKE_MUJOCO.step_error = None

    assert advanced is False
    assert runtime._running is True
    assert runtime._numerical_fault_latched is True
    assert runtime._runtime_fault["type"] == "numerical_instability"
    assert "mujoco_fatal_error" in runtime._runtime_fault["reasons"]
    assert writes == [True]
