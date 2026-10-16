"""Validated loading of persisted MuJoCo checkpoint pairs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MUJOCO_STATE_COMPONENT_BITS = tuple(1 << index for index in range(14))


def checkpoint_resync_lowcmd_status(
    last_seen_by_robot: Mapping[str, float | None],
    *,
    wall_now: float,
    lowcmd_timeout_s: float,
    minimum_wait_until: float,
    recovery_deadline: float,
) -> tuple[str, dict[str, float]]:
    """Decide whether frozen checkpoint physics can safely resume."""

    now = float(wall_now)
    timeout_s = max(0.0, float(lowcmd_timeout_s))
    stale = {
        str(robot_id): (
            float("inf")
            if last_seen is None
            else max(0.0, now - float(last_seen))
        )
        for robot_id, last_seen in last_seen_by_robot.items()
        if last_seen is None or now - float(last_seen) > timeout_s
    }
    if now < float(minimum_wait_until):
        return "minimum_wait", stale
    if not stale:
        return "ready", {}
    if now < float(recovery_deadline):
        return "waiting_lowcmd", stale
    return "fault", stale


def select_mujoco_state_components(
    state: np.ndarray,
    *,
    source_spec: int,
    target_spec: int,
    component_size: Any,
) -> np.ndarray:
    """Extract a safe state subset from MuJoCo's bit-ordered flat buffer."""

    source_spec = int(source_spec)
    target_spec = int(target_spec)
    if target_spec & ~source_spec:
        raise ValueError(
            f"target state spec {target_spec} is not a subset of source spec {source_spec}"
        )
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    offset = 0
    selected: list[np.ndarray] = []
    for bit in MUJOCO_STATE_COMPONENT_BITS:
        if not source_spec & bit:
            continue
        size = int(component_size(bit))
        if size < 0 or offset + size > state.size:
            raise ValueError(f"invalid MuJoCo state component size for bit {bit}: {size}")
        component = state[offset : offset + size]
        if target_spec & bit:
            selected.append(component)
        offset += size
    if offset != state.size:
        raise ValueError(
            f"MuJoCo state buffer size {state.size} does not match decoded size {offset}"
        )
    if not selected:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(selected)


def enforce_free_joint_state(
    data: Any,
    *,
    qpos_adr: int,
    dof_adr: int,
    qpos_target: np.ndarray,
    qvel_target: np.ndarray,
) -> None:
    """Hold one MuJoCo free joint at an exact checkpoint state.

    This helper is used only during the bounded planner-resynchronization
    window after rollback.  It deliberately touches the free-root state, not
    any actuated joint, so SONIC still receives and controls the complete
    articulated posture while its recurrent/planner history is reset.
    """

    qpos_target = np.asarray(qpos_target, dtype=np.float64)
    qvel_target = np.asarray(qvel_target, dtype=np.float64)
    if qpos_target.shape != (7,) or qvel_target.shape != (6,):
        raise ValueError("free-joint checkpoint targets must have shapes (7,) and (6,)")
    data.qpos[qpos_adr : qpos_adr + 7] = qpos_target
    data.qvel[dof_adr : dof_adr + 6] = qvel_target
    if hasattr(data, "qacc"):
        data.qacc[dof_adr : dof_adr + 6] = 0.0
    if hasattr(data, "qacc_warmstart"):
        data.qacc_warmstart[dof_adr : dof_adr + 6] = 0.0


@dataclass(frozen=True)
class PersistedCheckpoint:
    checkpoint_id: str
    state: np.ndarray
    metadata: dict[str, Any]


def load_persisted_checkpoints(
    checkpoint_dir: Path | None,
) -> tuple[dict[str, PersistedCheckpoint], tuple[str, ...]]:
    """Load complete checkpoint pairs and report rejected artifacts."""

    if checkpoint_dir is None:
        return {}, ()
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return {}, ()

    loaded: dict[str, PersistedCheckpoint] = {}
    rejected: list[str] = []
    stems = {
        path.stem
        for suffix in (".json", ".npz")
        for path in checkpoint_dir.glob(f"*{suffix}")
        if path.is_file()
    }
    for checkpoint_id in sorted(stems):
        json_path = checkpoint_dir / f"{checkpoint_id}.json"
        state_path = checkpoint_dir / f"{checkpoint_id}.npz"
        if not _valid_checkpoint_id(checkpoint_id):
            rejected.append(f"{checkpoint_id}: invalid checkpoint id")
            continue
        if not json_path.is_file() or not state_path.is_file():
            rejected.append(f"{checkpoint_id}: incomplete JSON/NPZ pair")
            continue
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
            if metadata.get("checkpoint_id") != checkpoint_id:
                raise ValueError("metadata checkpoint_id does not match filename")
            state_spec = int(metadata["state_spec"])
            declared_size = int(metadata["state_size"])
            with np.load(state_path, allow_pickle=False) as archive:
                state = np.asarray(archive["state"], dtype=np.float64)
                archived_state_spec = int(np.asarray(archive["state_spec"]).reshape(-1)[0])
            if state.ndim != 1:
                raise ValueError("state array must be one-dimensional")
            if state.size != declared_size:
                raise ValueError(
                    f"state size {state.size} does not match metadata {declared_size}"
                )
            if archived_state_spec != state_spec:
                raise ValueError("NPZ state_spec does not match metadata")
            if not np.all(np.isfinite(state)):
                raise ValueError("state array contains non-finite values")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            rejected.append(f"{checkpoint_id}: {exc}")
            continue
        loaded[checkpoint_id] = PersistedCheckpoint(
            checkpoint_id=checkpoint_id,
            state=state,
            metadata=metadata,
        )
    return loaded, tuple(rejected)


def _valid_checkpoint_id(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and all(
        character.isalnum() or character in "-_." for character in value
    ) and len(value) <= 96
