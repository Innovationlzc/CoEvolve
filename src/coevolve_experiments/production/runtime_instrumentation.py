"""Read-only MuJoCo instrumentation for the frozen V83 runtime.

The post-Table-1 integration is not allowed to edit ``dual_runtime.py``.  This
module therefore provides a small mixin and a recorder that inspect the
already-computed ``MjData`` after a normal V83 state publication.  They never
write controls, call ``mj_step``, mutate checkpoints, or decide a terminal
outcome.

The recorder keeps samples from exactly one trusted checkpoint.  A checkpoint
change clears the rolling window so a terminal stability proof cannot span a
rollback or two accepted states.  Its output is the strict raw trace consumed
by :mod:`coevolve_experiments.reporting.terminal_evidence`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeVar

from ..reporting.terminal_evidence import (
    CONTACT_DISTANCE_TOLERANCE_M,
    MAX_UPRIGHT_ROBOT_TILT_DEG,
    MUJOCO_TERMINAL_SAMPLE_TRACE_KIND,
    MUJOCO_TERMINAL_SAMPLE_TRACE_SCHEMA_VERSION,
    MUJOCO_TERMINAL_STATE_KIND,
    MUJOCO_TERMINAL_STATE_SCHEMA_VERSION,
    PAYLOAD_HALF_HEIGHT_M,
    TARGET_SCALE_XYZ_M,
    TERMINAL_STABLE_WINDOW_S,
)
from ..serialization import to_primitive


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROBOT_IDS = ("front", "back")
_CONTACT_ROLES = frozenset(
    {"payload", "target", "floor", "wall", "front_robot", "back_robot"}
)


class RuntimeInstrumentationError(RuntimeError):
    """The runtime cannot provide an unambiguous raw evidence sample."""


@dataclass(frozen=True, slots=True)
class RuntimeTraceBinding:
    """Identity copied from consumed production channels plus live MuJoCo."""

    run_id: str
    seed: int
    scene_fingerprint: str
    scene_realization_sha256: str
    compiled_model_sha256: str
    scene_execution_evidence_fingerprint: str
    target_center_xyz_m: tuple[float, float, float]
    simulator_version: str
    contact_role_registry_source_sha256: str
    wall_prim_names: tuple[str, ...]
    target_geom_names: tuple[str, ...]
    floor_geom_names: tuple[str, ...]
    target_scale_xyz_m: tuple[float, float, float] = TARGET_SCALE_XYZ_M
    payload_half_height_m: float = PAYLOAD_HALF_HEIGHT_M

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("runtime trace run_id cannot be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("runtime trace seed must be a non-negative integer")
        for label, value in (
            ("scene_fingerprint", self.scene_fingerprint),
            ("scene_realization_sha256", self.scene_realization_sha256),
            ("compiled_model_sha256", self.compiled_model_sha256),
            (
                "scene_execution_evidence_fingerprint",
                self.scene_execution_evidence_fingerprint,
            ),
            (
                "contact_role_registry_source_sha256",
                self.contact_role_registry_source_sha256,
            ),
        ):
            if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"runtime trace {label} must be SHA-256")
        _finite_vector(self.target_center_xyz_m, 3, "target center")
        scale = _finite_vector(self.target_scale_xyz_m, 3, "target scale")
        if tuple(scale) != tuple(TARGET_SCALE_XYZ_M):
            raise ValueError("runtime trace target scale differs from the terminal contract")
        if not math.isclose(
            _finite_number(self.payload_half_height_m, "payload half height"),
            PAYLOAD_HALF_HEIGHT_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "runtime trace payload half height differs from the terminal contract"
            )
        if not isinstance(self.simulator_version, str) or not self.simulator_version.strip():
            raise ValueError("runtime trace simulator_version cannot be empty")
        for label, values in (
            ("wall prim names", self.wall_prim_names),
            ("target geom names", self.target_geom_names),
            ("floor geom names", self.floor_geom_names),
        ):
            _validated_role_names(values, label)

    @classmethod
    def from_consumed_channels(
        cls,
        term1_simulator_asset: Mapping[str, Any],
        formal_run_manifest: Mapping[str, Any],
        *,
        simulator_version: str,
    ) -> "RuntimeTraceBinding":
        """Build from the two relevant values returned by the injection gate.

        No provider credential, output path, or caller-authored identity is
        accepted here.  The production adapter must pass the mappings returned
        by ``ProductionInjectionConsumptionGate.consume``.
        """

        term1 = _mapping(term1_simulator_asset, "term1 simulator channel")
        manifest = _mapping(formal_run_manifest, "formal manifest channel")
        run = _mapping(manifest.get("run_spec"), "formal RunSpec")
        target = _mapping(term1.get("target"), "scene target")
        contact_roles = _mapping(
            term1.get("contact_role_registry"),
            "scene contact-role registry",
        )
        if set(contact_roles) != {
            "schema_version",
            "source_execution_sha256",
            "wall_prim_names",
            "target_geom_names",
            "floor_geom_names",
        } or contact_roles.get("schema_version") != 1:
            raise ValueError("scene contact-role registry fields differ")
        registry_source_sha256 = _sha(
            contact_roles.get("source_execution_sha256"),
            "contact-role registry source SHA-256",
        )
        execution_sha256 = _sha(
            term1.get("execution_sha256"),
            "scene execution SHA-256",
        )
        if registry_source_sha256 != execution_sha256:
            raise ValueError(
                "contact-role registry is not bound to the scene execution bytes"
            )
        center = target.get("center")
        # ``compile_production_injection_bundle`` serializes the scene Vec3 as
        # an ``{x,y,z}`` object.  Retain array support for already-recorded
        # channel fixtures, but normalize the production representation here.
        if isinstance(center, Mapping) and set(center) == {"x", "y", "z"}:
            center = (center["x"], center["y"], center["z"])
        if not isinstance(center, (tuple, list)):
            raise ValueError(
                "scene target center must be an {x,y,z} object or three-value array"
            )
        radius = _finite_number(target.get("radius_m"), "target radius")
        if not math.isclose(radius, TARGET_SCALE_XYZ_M[0] / 2.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("scene target radius differs from the terminal contract")
        return cls(
            run_id=_nonempty(run.get("run_id"), "RunSpec run_id"),
            seed=_integer(run.get("seed"), "RunSpec seed"),
            scene_fingerprint=_sha(run.get("scene_fingerprint"), "scene fingerprint"),
            scene_realization_sha256=_sha(
                term1.get("execution_composite_sha256"),
                "scene realization SHA-256",
            ),
            compiled_model_sha256=_sha(
                term1.get("compiled_model_sha256"),
                "compiled model SHA-256",
            ),
            scene_execution_evidence_fingerprint=_sha(
                term1.get("execution_evidence_fingerprint"),
                "scene evidence fingerprint",
            ),
            target_center_xyz_m=tuple(
                _finite_vector(center, 3, "target center")
            ),
            simulator_version=simulator_version,
            contact_role_registry_source_sha256=registry_source_sha256,
            wall_prim_names=_validated_role_names(
                contact_roles.get("wall_prim_names"),
                "wall prim names",
            ),
            target_geom_names=_validated_role_names(
                contact_roles.get("target_geom_names"),
                "target geom names",
            ),
            floor_geom_names=_validated_role_names(
                contact_roles.get("floor_geom_names"),
                "floor geom names",
            ),
        )

    @property
    def simulator(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "name": "MuJoCo",
                "version": self.simulator_version,
                "compiled_model_sha256": self.compiled_model_sha256,
                "scene_realization_sha256": self.scene_realization_sha256,
                "scene_fingerprint": self.scene_fingerprint,
                "scene_execution_evidence_fingerprint": (
                    self.scene_execution_evidence_fingerprint
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class TerminalEvidenceDocuments:
    """Raw trace and deterministic summary passed to the evidence writer."""

    terminal_sample_trace: Mapping[str, Any]
    terminal_state: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CapturedSample:
    checkpoint_id: str
    checkpoint_event_id: str
    time_s: float
    sample: Mapping[str, Any]
    lifecycle: Mapping[str, bool]


class MuJoCoTerminalTraceRecorder:
    """Collect an immutable-view rolling trace without changing simulation."""

    def __init__(
        self,
        binding: RuntimeTraceBinding,
        *,
        sample_interval_s: float = 0.10,
        retention_window_s: float = 4.0,
    ) -> None:
        if not isinstance(binding, RuntimeTraceBinding):
            raise TypeError("binding must be RuntimeTraceBinding")
        interval = _finite_number(sample_interval_s, "sample interval")
        retention = _finite_number(retention_window_s, "retention window")
        if interval <= 0.0:
            raise ValueError("sample interval must be positive")
        if retention + 1e-12 < TERMINAL_STABLE_WINDOW_S:
            raise ValueError("retention window must cover the terminal stability window")
        self.binding = binding
        self.sample_interval_s = interval
        self.retention_window_s = retention
        self._samples: list[_CapturedSample] = []
        self._active_checkpoint_id = ""
        self._active_checkpoint_event_id = ""
        self._last_time_s: float | None = None
        self._lock = RLock()

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def checkpoint_id(self) -> str:
        with self._lock:
            return self._active_checkpoint_id

    @property
    def checkpoint_event_id(self) -> str:
        with self._lock:
            return self._active_checkpoint_event_id

    def capture(self, runtime: Any, *, force: bool = False) -> bool:
        """Read one post-step state; return whether a sample was appended."""

        time_s = _finite_number(getattr(getattr(runtime, "data", None), "time", None), "MuJoCo time")
        checkpoint_id, checkpoint_event_id = _runtime_checkpoint_identity(runtime)
        with self._lock:
            if (
                checkpoint_id != self._active_checkpoint_id
                or checkpoint_event_id != self._active_checkpoint_event_id
            ):
                self._samples.clear()
                self._last_time_s = None
                self._active_checkpoint_id = checkpoint_id
                self._active_checkpoint_event_id = checkpoint_event_id
            if not checkpoint_id:
                return False
            if self._last_time_s is not None:
                if time_s < self._last_time_s:
                    raise RuntimeInstrumentationError(
                        "MuJoCo time regressed without a checkpoint identity change"
                    )
                if time_s == self._last_time_s:
                    return False
                if not force and time_s - self._last_time_s < self.sample_interval_s:
                    return False
            sample, lifecycle = _capture_runtime_sample(
                runtime,
                time_s,
                binding=self.binding,
            )
            self._samples.append(
                _CapturedSample(
                    checkpoint_id,
                    checkpoint_event_id,
                    time_s,
                    sample,
                    lifecycle,
                )
            )
            self._last_time_s = time_s
            cutoff = time_s - self.retention_window_s
            while len(self._samples) > 1 and self._samples[0].time_s < cutoff:
                self._samples.pop(0)
            return True

    def build_terminal_documents(
        self,
        *,
        checkpoint_id: str | None = None,
        require_stable_window: bool = False,
    ) -> TerminalEvidenceDocuments:
        """Freeze the current single-checkpoint window into schema documents."""

        with self._lock:
            selected_checkpoint = checkpoint_id or self._active_checkpoint_id
            selected_event = self._active_checkpoint_event_id
            if not isinstance(selected_checkpoint, str) or not selected_checkpoint.strip():
                raise RuntimeInstrumentationError("terminal checkpoint ID is unavailable")
            if selected_checkpoint != self._active_checkpoint_id:
                raise RuntimeInstrumentationError(
                    "requested terminal checkpoint differs from the sampled checkpoint"
                )
            samples = tuple(self._samples)
        if len(samples) < 2:
            raise RuntimeInstrumentationError("terminal trace requires at least two samples")
        if any(item.checkpoint_id != selected_checkpoint for item in samples):
            raise RuntimeInstrumentationError("terminal trace crosses checkpoint identities")
        if not selected_event or any(
            item.checkpoint_event_id != selected_event for item in samples
        ):
            raise RuntimeInstrumentationError(
                "terminal trace crosses checkpoint restore/save epochs"
            )
        window = samples[-1].time_s - samples[0].time_s
        if require_stable_window and window + 1e-9 < TERMINAL_STABLE_WINDOW_S:
            raise RuntimeInstrumentationError(
                "terminal trace does not cover the three-second stability window"
            )
        trace_samples = [
            {"sample_index": index, **dict(item.sample)}
            for index, item in enumerate(samples)
        ]
        lifecycle = dict(samples[-1].lifecycle)
        trace = {
            "kind": MUJOCO_TERMINAL_SAMPLE_TRACE_KIND,
            "schema_version": MUJOCO_TERMINAL_SAMPLE_TRACE_SCHEMA_VERSION,
            "run_id": self.binding.run_id,
            "seed": self.binding.seed,
            "checkpoint_id": selected_checkpoint,
            "checkpoint_event_id": selected_event,
            "simulator": dict(self.binding.simulator),
            "target": {
                "center_xyz_m": list(self.binding.target_center_xyz_m),
                "scale_xyz_m": list(self.binding.target_scale_xyz_m),
            },
            "payload_geometry": {
                "half_height_m": self.binding.payload_half_height_m,
            },
            "sample_window_s": window,
            "samples": trace_samples,
            "lifecycle": lifecycle,
        }
        state = _derive_terminal_state(trace)
        return TerminalEvidenceDocuments(
            terminal_sample_trace=MappingProxyType(trace),
            terminal_state=MappingProxyType(state),
        )

    def write_live_snapshot(self, path: str | Path) -> Path:
        """Atomically refresh a non-evidence rolling snapshot for Term4."""

        documents = self.build_terminal_documents(require_stable_window=False)
        destination = _safe_absolute_snapshot_path(path)
        payload = _json_bytes(
            {
                "kind": "coevolve_mujoco_live_terminal_snapshot",
                "schema_version": 1,
                "paper_evidence": False,
                "terminal_sample_trace": documents.terminal_sample_trace,
                "terminal_state": documents.terminal_state,
            }
        )
        _atomic_replace_snapshot(destination, payload)
        return destination


class InstrumentedRuntimeMixin:
    """Mixin inserted ahead of the frozen runtime by the Term1 wrapper.

    Sampling is attached to V83's existing low-frequency state publication,
    not its high-frequency control step.  Consequently the physics/control
    path and the number or timing of ``mj_step`` calls remain unchanged.
    """

    def __init__(
        self,
        *args: Any,
        terminal_trace_recorder: MuJoCoTerminalTraceRecorder,
        terminal_trace_snapshot_path: str | Path,
        **kwargs: Any,
    ) -> None:
        if not isinstance(terminal_trace_recorder, MuJoCoTerminalTraceRecorder):
            raise TypeError("terminal_trace_recorder has the wrong type")
        snapshot = _safe_absolute_snapshot_path(terminal_trace_snapshot_path)
        self._coevolve_terminal_trace_recorder = terminal_trace_recorder
        self._coevolve_terminal_trace_snapshot_path = snapshot
        super().__init__(*args, **kwargs)

    def _write_runtime_state(self) -> None:
        super()._write_runtime_state()
        recorder = self._coevolve_terminal_trace_recorder
        if recorder.capture(self):
            try:
                recorder.write_live_snapshot(
                    self._coevolve_terminal_trace_snapshot_path
                )
            except RuntimeInstrumentationError:
                # One initial sample cannot yet form a trace.  Any other
                # instrumentation error is deliberately not swallowed.
                if recorder.sample_count >= 2:
                    raise


_RuntimeType = TypeVar("_RuntimeType", bound=type)


def instrument_runtime_class(base_class: _RuntimeType) -> _RuntimeType:
    """Return the reviewed external subclass for one pinned V83 base class."""

    if not isinstance(base_class, type) or not callable(
        getattr(base_class, "_write_runtime_state", None)
    ):
        raise TypeError("base runtime must expose _write_runtime_state")
    return type(
        "InstrumentedDualG1SonicMujocoRuntime",
        (InstrumentedRuntimeMixin, base_class),
        {
            "__module__": __name__,
            "__doc__": (
                "External evidence-only subclass of the byte-pinned V83 runtime."
            ),
        },
    )


def load_live_terminal_snapshot(path: str | Path) -> TerminalEvidenceDocuments:
    """Strictly load the latest Term1 snapshot for final publication."""

    source = _safe_absolute_snapshot_path(path)
    if source.is_symlink() or not source.is_file():
        raise RuntimeInstrumentationError("live terminal snapshot is unavailable")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeInstrumentationError("live terminal snapshot is invalid") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "kind",
        "schema_version",
        "paper_evidence",
        "terminal_sample_trace",
        "terminal_state",
    }:
        raise RuntimeInstrumentationError("live terminal snapshot fields differ")
    if (
        value.get("kind") != "coevolve_mujoco_live_terminal_snapshot"
        or value.get("schema_version") != 1
        or value.get("paper_evidence") is not False
    ):
        raise RuntimeInstrumentationError("live terminal snapshot identity differs")
    trace = _mapping(value.get("terminal_sample_trace"), "terminal trace")
    state = _mapping(value.get("terminal_state"), "terminal state")
    # Re-derive the summary so a partially written or edited live file cannot
    # smuggle aggregate values into the immutable evidence writer.
    derived = _derive_terminal_state(trace)
    if dict(state) != derived:
        raise RuntimeInstrumentationError("live terminal summary differs from raw samples")
    return TerminalEvidenceDocuments(
        terminal_sample_trace=MappingProxyType(dict(trace)),
        terminal_state=MappingProxyType(dict(state)),
    )


def _capture_runtime_sample(
    runtime: Any,
    time_s: float,
    *,
    binding: RuntimeTraceBinding,
) -> tuple[dict[str, Any], dict[str, bool]]:
    data = getattr(runtime, "data", None)
    model = getattr(runtime, "model", None)
    if data is None or model is None:
        raise RuntimeInstrumentationError("runtime has no MuJoCo model/data")
    payload_geom_id = _nonnegative_integer(
        getattr(runtime, "payload_geom_id", None), "payload geom ID"
    )
    payload_qpos_adr = _nonnegative_integer(
        getattr(runtime, "payload_qpos_adr", None), "payload qpos address"
    )
    qpos = getattr(data, "qpos", None)
    payload_pose = _pose_from_qpos(qpos, payload_qpos_adr, "payload")
    robot_poses: dict[str, Any] = {}
    slots = tuple(getattr(runtime, "slots", ()))
    for slot in slots:
        spec = getattr(slot, "spec", None)
        robot_id = getattr(spec, "robot_id", None)
        if robot_id not in _ROBOT_IDS or robot_id in robot_poses:
            raise RuntimeInstrumentationError("runtime robot slots are not exact front/back")
        pelvis_body_id = _nonnegative_integer(
            getattr(slot, "pelvis_body_id", None), f"{robot_id} pelvis body ID"
        )
        robot_poses[str(robot_id)] = _pose_from_body_state(
            data,
            pelvis_body_id,
            f"{robot_id} pelvis",
        )
    if set(robot_poses) != set(_ROBOT_IDS):
        raise RuntimeInstrumentationError("runtime must expose exactly front/back slots")

    contacts = []
    ncon = _nonnegative_integer(getattr(data, "ncon", None), "contact count")
    contact_array = getattr(data, "contact", None)
    for index in range(ncon):
        contact = contact_array[index]
        geom1 = _nonnegative_integer(getattr(contact, "geom1", None), "contact geom1")
        geom2 = _nonnegative_integer(getattr(contact, "geom2", None), "contact geom2")
        if payload_geom_id not in (geom1, geom2):
            continue
        role1 = _geom_role(runtime, geom1, payload_geom_id, binding)
        role2 = _geom_role(runtime, geom2, payload_geom_id, binding)
        if role1 == role2:
            raise RuntimeInstrumentationError("payload contact roles are ambiguous")
        contacts.append(
            {
                "geom1_id": geom1,
                "geom2_id": geom2,
                "geom1_role": role1,
                "geom2_role": role2,
                "distance_m": _finite_number(
                    getattr(contact, "dist", None), "contact distance"
                ),
            }
        )
    lifecycle = {
        "release_completed": _strict_runtime_bool(
            runtime, "_payload_release_active"
        ),
        "retreat_completed": _strict_runtime_bool(
            runtime, "_post_release_crouched_retreat_completed"
        ),
        "stand_recovered": _strict_runtime_bool(
            runtime, "_post_release_stand_recovered"
        ),
    }
    sample = {
        "time_s": time_s,
        "payload_released": lifecycle["release_completed"],
        "payload_pose": payload_pose,
        "robot_poses": robot_poses,
        "contacts": contacts,
    }
    return sample, lifecycle


def _runtime_checkpoint_identity(runtime: Any) -> tuple[str, str]:
    status = getattr(runtime, "_checkpoint_status", None)
    if not isinstance(status, Mapping):
        raise RuntimeInstrumentationError("runtime checkpoint status is unavailable")
    value = status.get("checkpoint_id")
    if not isinstance(value, str):
        raise RuntimeInstrumentationError("runtime checkpoint ID has the wrong type")
    event_id = status.get("event_id")
    if not isinstance(event_id, str):
        raise RuntimeInstrumentationError(
            "runtime checkpoint event ID has the wrong type"
        )
    checkpoint_id = value.strip()
    checkpoint_event_id = event_id.strip()
    if checkpoint_id and not checkpoint_event_id:
        raise RuntimeInstrumentationError(
            "runtime checkpoint event ID is unavailable"
        )
    return checkpoint_id, checkpoint_event_id


def _pose_from_qpos(qpos: Any, address: int, label: str) -> dict[str, list[float]]:
    try:
        values = tuple(float(value) for value in qpos[address : address + 7])
    except (TypeError, ValueError) as exc:
        raise RuntimeInstrumentationError(f"{label} pose is unreadable") from exc
    if len(values) != 7 or any(not math.isfinite(value) for value in values):
        raise RuntimeInstrumentationError(f"{label} pose is incomplete or non-finite")
    quaternion = values[3:]
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeInstrumentationError(f"{label} quaternion is not unit length")
    return {
        "position_xyz_m": list(values[:3]),
        "quaternion_wxyz": list(quaternion),
    }


def _pose_from_body_state(
    data: Any,
    body_id: int,
    label: str,
) -> dict[str, list[float]]:
    """Read the actual pelvis body frame used by the frozen V83 verifier."""

    try:
        position = tuple(float(value) for value in data.xpos[body_id])
        quaternion = tuple(float(value) for value in data.xquat[body_id])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeInstrumentationError(f"{label} pose is unreadable") from exc
    if len(position) != 3 or len(quaternion) != 4 or any(
        not math.isfinite(value) for value in (*position, *quaternion)
    ):
        raise RuntimeInstrumentationError(
            f"{label} pose is incomplete or non-finite"
        )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeInstrumentationError(f"{label} quaternion is not unit length")
    return {
        "position_xyz_m": list(position),
        "quaternion_wxyz": list(quaternion),
    }


def _geom_role(
    runtime: Any,
    geom_id: int,
    payload_geom_id: int,
    binding: RuntimeTraceBinding,
) -> str:
    if geom_id == payload_geom_id:
        return "payload"
    model = runtime.model
    try:
        geom_name = str(model.geom(geom_id).name or "")
        body_id = int(model.geom_bodyid[geom_id])
        body_name = str(model.body(body_id).name or "")
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeInstrumentationError("contact geometry identity is unreadable") from exc
    target_id = getattr(runtime, "target_geom_id", None)
    if geom_id == target_id or _matches_registered_name(
        geom_name,
        binding.target_geom_names,
    ):
        return "target"
    if _matches_registered_name(geom_name, binding.floor_geom_names) or _matches_registered_name(
        body_name,
        binding.floor_geom_names,
    ):
        return "floor"
    if _matches_registered_name(geom_name, binding.wall_prim_names) or _matches_registered_name(
        body_name,
        binding.wall_prim_names,
    ):
        return "wall"
    for robot_id in _ROBOT_IDS:
        prefix = f"{robot_id}_"
        if geom_name.startswith(prefix) or body_name.startswith(prefix):
            return f"{robot_id}_robot"
    raise RuntimeInstrumentationError(
        f"payload contact has no reviewed role: geom={geom_name!r}, body={body_name!r}"
    )


def _matches_registered_name(value: str, names: tuple[str, ...]) -> bool:
    return any(value == name or value.startswith(f"{name}_") for name in names)


def _derive_terminal_state(trace: Mapping[str, Any]) -> dict[str, Any]:
    samples = trace.get("samples")
    target = _mapping(trace.get("target"), "trace target")
    geometry = _mapping(trace.get("payload_geometry"), "payload geometry")
    if not isinstance(samples, list) or len(samples) < 2:
        raise RuntimeInstrumentationError("terminal trace requires at least two samples")
    center = _finite_vector(target.get("center_xyz_m"), 3, "target center")
    scale = _finite_vector(target.get("scale_xyz_m"), 3, "target scale")
    half_height = _finite_number(geometry.get("half_height_m"), "payload half height")
    expected_z = center[2] + abs(scale[2]) / 2.0 + half_height
    xy_errors: list[float] = []
    z_errors: list[float] = []
    payload_tilts: list[float] = []
    released: list[bool] = []
    contact_counts = {role: 0 for role in ("target", "floor", "wall", "robot")}
    robot_heights = {robot_id: [] for robot_id in _ROBOT_IDS}
    robot_distances = {robot_id: [] for robot_id in _ROBOT_IDS}
    robot_tilts = {robot_id: [] for robot_id in _ROBOT_IDS}
    times: list[float] = []
    for expected_index, raw in enumerate(samples):
        sample = _mapping(raw, f"trace sample {expected_index}")
        if sample.get("sample_index") != expected_index:
            raise RuntimeInstrumentationError("terminal sample indices are not contiguous")
        time_s = _finite_number(sample.get("time_s"), "terminal sample time")
        if times and time_s <= times[-1]:
            raise RuntimeInstrumentationError("terminal sample times are not increasing")
        times.append(time_s)
        if not isinstance(sample.get("payload_released"), bool):
            raise RuntimeInstrumentationError("payload release sample is not boolean")
        released.append(sample["payload_released"])
        payload_position, payload_tilt = _validated_pose(
            sample.get("payload_pose"), "payload"
        )
        xy_errors.append(math.hypot(payload_position[0] - center[0], payload_position[1] - center[1]))
        z_errors.append(abs(payload_position[2] - expected_z))
        payload_tilts.append(payload_tilt)
        robot_poses = _mapping(sample.get("robot_poses"), "robot poses")
        if set(robot_poses) != set(_ROBOT_IDS):
            raise RuntimeInstrumentationError("terminal sample robot roles differ")
        for robot_id in _ROBOT_IDS:
            position, tilt = _validated_pose(robot_poses[robot_id], robot_id)
            robot_heights[robot_id].append(position[2])
            robot_distances[robot_id].append(
                math.hypot(position[0] - center[0], position[1] - center[1])
            )
            robot_tilts[robot_id].append(tilt)
        roles = _active_payload_contact_roles(sample.get("contacts"))
        contact_counts["target"] += "target" in roles
        contact_counts["floor"] += "floor" in roles
        contact_counts["wall"] += "wall" in roles
        contact_counts["robot"] += bool(
            {"front_robot", "back_robot"}.intersection(roles)
        )
    lifecycle = _mapping(trace.get("lifecycle"), "terminal lifecycle")
    if set(lifecycle) != {"release_completed", "retreat_completed", "stand_recovered"}:
        raise RuntimeInstrumentationError("terminal lifecycle fields differ")
    if any(not isinstance(lifecycle[key], bool) for key in lifecycle):
        raise RuntimeInstrumentationError("terminal lifecycle values must be boolean")
    window = times[-1] - times[0]
    declared_window = _finite_number(trace.get("sample_window_s"), "sample window")
    if not math.isclose(window, declared_window, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeInstrumentationError("terminal sample window differs")
    return {
        "kind": MUJOCO_TERMINAL_STATE_KIND,
        "schema_version": MUJOCO_TERMINAL_STATE_SCHEMA_VERSION,
        "run_id": trace.get("run_id"),
        "seed": trace.get("seed"),
        "checkpoint_id": trace.get("checkpoint_id"),
        "checkpoint_event_id": trace.get("checkpoint_event_id"),
        "simulator": dict(_mapping(trace.get("simulator"), "simulator identity")),
        "sample_window_s": window,
        "sample_count": len(samples),
        "payload": {
            "released": all(released),
            "target_contact_all_samples": contact_counts["target"] == len(samples),
            "floor_contact_samples": contact_counts["floor"],
            "wall_contact_samples": contact_counts["wall"],
            "robot_contact_samples": contact_counts["robot"],
            "xy_error_m": max(xy_errors),
            "z_error_m": max(z_errors),
            "tilt_deg": max(payload_tilts),
        },
        "robots": {
            robot_id: {
                "upright": max(robot_tilts[robot_id]) <= MAX_UPRIGHT_ROBOT_TILT_DEG,
                "pelvis_height_m": min(robot_heights[robot_id]),
                "distance_to_target_m": min(robot_distances[robot_id]),
            }
            for robot_id in _ROBOT_IDS
        },
        "lifecycle": dict(lifecycle),
    }


def _validated_pose(value: object, label: str) -> tuple[tuple[float, ...], float]:
    pose = _mapping(value, f"{label} pose")
    if set(pose) != {"position_xyz_m", "quaternion_wxyz"}:
        raise RuntimeInstrumentationError(f"{label} pose fields differ")
    position = _finite_vector(pose.get("position_xyz_m"), 3, f"{label} position")
    quaternion = _finite_vector(pose.get("quaternion_wxyz"), 4, f"{label} quaternion")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise RuntimeInstrumentationError(f"{label} quaternion is not unit length")
    _, x, y, _ = (value / norm for value in quaternion)
    up_dot = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return position, math.degrees(math.acos(up_dot))


def _active_payload_contact_roles(value: object) -> set[str]:
    if not isinstance(value, list):
        raise RuntimeInstrumentationError("terminal contacts must be an array")
    result: set[str] = set()
    for item in value:
        contact = _mapping(item, "terminal contact")
        if set(contact) != {
            "geom1_id",
            "geom2_id",
            "geom1_role",
            "geom2_role",
            "distance_m",
        }:
            raise RuntimeInstrumentationError("terminal contact fields differ")
        geom1 = _nonnegative_integer(contact.get("geom1_id"), "contact geom1")
        geom2 = _nonnegative_integer(contact.get("geom2_id"), "contact geom2")
        role1, role2 = contact.get("geom1_role"), contact.get("geom2_role")
        if geom1 == geom2 or role1 not in _CONTACT_ROLES or role2 not in _CONTACT_ROLES or role1 == role2:
            raise RuntimeInstrumentationError("terminal contact identity is invalid")
        distance = _finite_number(contact.get("distance_m"), "contact distance")
        if distance <= CONTACT_DISTANCE_TOLERANCE_M and "payload" in {role1, role2}:
            result.update({str(role1), str(role2)} - {"payload"})
    return result


def _strict_runtime_bool(runtime: Any, name: str) -> bool:
    value = getattr(runtime, name, None)
    if not isinstance(value, bool):
        raise RuntimeInstrumentationError(f"runtime lifecycle field {name} is unavailable")
    return value


def _safe_absolute_snapshot_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeInstrumentationError("live snapshot path must be absolute")
    if path.name in {"", ".", ".."}:
        raise RuntimeInstrumentationError("live snapshot path is invalid")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeInstrumentationError("live snapshot parent must be a real directory")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise RuntimeInstrumentationError("live snapshot target must be a regular file")
    return path


def _atomic_replace_snapshot(path: Path, payload: bytes) -> None:
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                to_primitive(value),
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeInstrumentationError("live terminal snapshot is not JSON safe") from exc


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeInstrumentationError(f"{label} must be an object")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} cannot be empty")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be SHA-256")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _validated_role_names(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"runtime trace {label} must be a non-empty sequence")
    result = tuple(value)
    if any(
        not isinstance(item, str)
        or not item
        or item.strip() != item
        or "/" in item
        or "\\" in item
        for item in result
    ):
        raise ValueError(f"runtime trace {label} contains an invalid name")
    if len(result) != len(set(result)) or result != tuple(sorted(result)):
        raise ValueError(
            f"runtime trace {label} must be unique and lexicographically sorted"
        )
    return result


def _nonnegative_integer(value: object, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeInstrumentationError(f"{label} is unavailable") from exc
    if isinstance(value, bool) or parsed < 0 or parsed != value:
        raise RuntimeInstrumentationError(f"{label} is invalid")
    return parsed


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_vector(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    return tuple(_finite_number(item, label) for item in value)


__all__ = [
    "InstrumentedRuntimeMixin",
    "MuJoCoTerminalTraceRecorder",
    "RuntimeInstrumentationError",
    "RuntimeTraceBinding",
    "TerminalEvidenceDocuments",
    "instrument_runtime_class",
    "load_live_terminal_snapshot",
]
