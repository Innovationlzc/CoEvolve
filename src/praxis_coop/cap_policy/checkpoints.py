"""Control-plane client for MuJoCo attempt checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Literal

from .contracts import write_json


CheckpointAction = Literal["save", "restore"]


class CheckpointBackendFatalError(ConnectionError):
    """Raised when the live simulator disappears during a checkpoint request."""


@dataclass(frozen=True)
class CheckpointResult:
    action: CheckpointAction
    checkpoint_id: str
    event_id: str
    success: bool
    requested_at_unix: float
    acknowledged_at_unix: float
    runtime_state_before: dict[str, Any]
    runtime_state_after: dict[str, Any]
    error: str = ""


@dataclass
class SimulatorCheckpointClient:
    """Ask the live Term1 runtime to save or restore full physics state."""

    heartbeat_file: Path
    runtime_state_file: Path
    artifact_dir: Path
    timeout_s: float = 10.0
    poll_s: float = 0.05
    resend_s: float = 0.25
    restore_settle_s: float = 4.0
    runtime_state_max_age_s: float = 2.0
    backend_stale_grace_s: float = 0.5

    def save(self, checkpoint_id: str, *, reset_metrics: bool = False) -> CheckpointResult:
        return self._request("save", checkpoint_id, reset_metrics=reset_metrics)

    def restore(
        self,
        checkpoint_id: str,
        *,
        hold_planning: bool = False,
    ) -> CheckpointResult:
        return self._request(
            "restore",
            checkpoint_id,
            hold_planning=hold_planning,
        )

    def _request(
        self,
        action: CheckpointAction,
        checkpoint_id: str,
        *,
        reset_metrics: bool = False,
        hold_planning: bool = False,
    ) -> CheckpointResult:
        checkpoint_id = _validate_checkpoint_id(checkpoint_id)
        requested_at = time.time()
        event_id = f"{action}_{checkpoint_id}_{time.time_ns()}"
        before = _read_json(self.runtime_state_file)
        expected_runtime_instance_id = str(before.get("runtime_instance_id", "")).strip()
        initial_state_sequence = _nonnegative_int(before.get("state_sequence"))
        heartbeat = {
            "phase": f"sim_checkpoint_{action}",
            "time_unix": requested_at,
            "checkpoint_action": action,
            "checkpoint_id": checkpoint_id,
            "checkpoint_event_id": event_id,
        }
        if action == "restore":
            heartbeat["checkpoint_settle_s"] = max(0.0, float(self.restore_settle_s))
            heartbeat["checkpoint_hold_planning"] = bool(hold_planning)
        elif reset_metrics:
            heartbeat["checkpoint_reset_metrics"] = True
        _atomic_write_json(self.heartbeat_file, heartbeat)

        deadline = time.monotonic() + max(0.1, float(self.timeout_s))
        resend_interval = max(float(self.poll_s), float(self.resend_s), 0.01)
        next_resend = time.monotonic() + resend_interval
        after: dict[str, Any] = {}
        status: dict[str, Any] = {}
        backend_error = ""
        stale_since: float | None = None
        while time.monotonic() < deadline:
            after = _read_json(self.runtime_state_file)
            candidate = after.get("checkpoint", {}) if isinstance(after, dict) else {}
            status = candidate if isinstance(candidate, dict) else {}
            if status.get("event_id") == event_id:
                break
            now = time.monotonic()
            backend_error = _checkpoint_backend_error(
                after,
                action=action,
                expected_runtime_instance_id=expected_runtime_instance_id,
                initial_state_sequence=initial_state_sequence,
                max_age_s=max(0.1, float(self.runtime_state_max_age_s)),
            )
            if backend_error:
                if stale_since is None:
                    stale_since = now
                if now - stale_since >= max(0.0, float(self.backend_stale_grace_s)):
                    status = {"success": False, "error": backend_error}
                    break
            else:
                stale_since = None
            if now >= next_resend:
                heartbeat["time_unix"] = time.time()
                _atomic_write_json(self.heartbeat_file, heartbeat)
                next_resend = now + resend_interval
            time.sleep(max(0.01, float(self.poll_s)))
        else:
            status = {
                "success": False,
                "error": f"timeout waiting for Term1 checkpoint acknowledgement at {self.runtime_state_file}",
            }

        # A checkpoint request is edge-triggered. Leaving its event id in the
        # shared heartbeat lets a newly restarted Term1 replay an already
        # completed or timed-out restore before SONIC LowCmd is live. Retain a
        # fresh neutral heartbeat instead; the runtime-state acknowledgement
        # and event artifact remain the durable record.
        _atomic_write_json(
            self.heartbeat_file,
            {
                "phase": "sim_checkpoint_event_cleared",
                "time_unix": time.time(),
                "last_checkpoint_action": action,
                "last_checkpoint_id": checkpoint_id,
                "last_checkpoint_event_id": event_id,
            },
        )

        result = CheckpointResult(
            action=action,
            checkpoint_id=checkpoint_id,
            event_id=event_id,
            success=bool(status.get("success", False)),
            requested_at_unix=requested_at,
            acknowledged_at_unix=time.time(),
            runtime_state_before=before,
            runtime_state_after=after,
            error=str(status.get("error", "")),
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.artifact_dir / f"{action}_{checkpoint_id}_{event_id}.json", result)
        if not result.success:
            if backend_error:
                raise CheckpointBackendFatalError(
                    f"MuJoCo checkpoint {action} failed for {checkpoint_id}: "
                    f"{result.error or backend_error}"
                )
            raise RuntimeError(
                f"MuJoCo checkpoint {action} failed for {checkpoint_id}: {result.error or 'no acknowledgement'}"
            )
        return result


def _validate_checkpoint_id(value: str) -> str:
    value = str(value).strip()
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in value):
        raise ValueError(f"invalid checkpoint_id: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"invalid checkpoint_id: {value!r}")
    return value[:96]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _checkpoint_backend_error(
    state: dict[str, Any],
    *,
    action: CheckpointAction,
    expected_runtime_instance_id: str,
    initial_state_sequence: int | None,
    max_age_s: float,
) -> str:
    """Return a fatal reason only when live-state evidence says Term1 is gone."""

    if not state:
        return "Term1 runtime state is missing while awaiting checkpoint acknowledgement"
    instance_id = str(state.get("runtime_instance_id", "")).strip()
    if expected_runtime_instance_id and instance_id and instance_id != expected_runtime_instance_id:
        return (
            "Term1 runtime instance changed while awaiting checkpoint acknowledgement: "
            f"expected={expected_runtime_instance_id} observed={instance_id}"
        )
    runtime_status = str(state.get("runtime_status", "")).strip()
    if runtime_status and runtime_status != "running":
        return f"Term1 runtime stopped while awaiting checkpoint acknowledgement: {runtime_status}"
    fault = state.get("runtime_fault", {})
    fault = fault if isinstance(fault, dict) else {}
    fault_type = str(fault.get("type", "")).strip()
    if fault_type and not (action == "restore" and fault_type == "numerical_instability"):
        return f"Term1 runtime reported fatal fault {fault_type}"
    try:
        age_s = max(0.0, time.time() - float(state.get("time_unix", 0.0)))
    except (TypeError, ValueError):
        age_s = float("inf")
    if age_s > max_age_s:
        return (
            "Term1 runtime state stopped advancing while awaiting checkpoint "
            f"acknowledgement: age={age_s:.2f}s limit={max_age_s:.2f}s"
        )
    sequence = _nonnegative_int(state.get("state_sequence"))
    if (
        initial_state_sequence is not None
        and sequence is not None
        and sequence < initial_state_sequence
    ):
        return (
            "Term1 runtime state sequence regressed while awaiting checkpoint "
            f"acknowledgement: initial={initial_state_sequence} observed={sequence}"
        )
    return ""


def _nonnegative_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".checkpoint.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)
