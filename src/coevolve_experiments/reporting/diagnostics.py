"""Collectors for Appendix A4 replay and A5 motor-calibration diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def collect_replay_diagnostics(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Normalize replay summaries without treating them as task-success runs."""

    rows: list[dict[str, Any]] = []
    for source in paths:
        path = Path(source)
        payload = _read_json(path)
        if not payload:
            continue
        mode = str(
            payload.get("planner_history_mode")
            or payload.get("controller_history")
            or payload.get("replay_mode")
            or ""
        )
        rows.append(
            {
                "source": str(path),
                "controller_history": mode,
                "full_success": bool(
                    payload.get("success", payload.get("full_success", False))
                ),
                "safe_replay_prefix": _first(
                    payload,
                    "safe_replay_prefix",
                    "safe_prefix_length",
                    "completed_actions",
                ),
                "total_actions": _first(
                    payload, "total_actions", "manifest_action_count"
                ),
                "failed_step": _first(
                    payload, "failed_step", "failure_step", "first_failed_step"
                ),
                "failure": str(
                    payload.get("failure")
                    or payload.get("failure_reason")
                    or payload.get("terminal_reason")
                    or ""
                ),
                "maximum_trajectory_divergence_m": _number(
                    _first(
                        payload,
                        "maximum_trajectory_divergence_m",
                        "max_endpoint_divergence_m",
                        "max_trajectory_divergence_m",
                    )
                ),
            }
        )
    return rows


def collect_skill_calibration(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Normalize linear and pivot calibration rows for Appendix A5."""

    rows: list[dict[str, Any]] = []
    for source in paths:
        path = Path(source)
        payload = _read_json(path)
        if not payload:
            continue
        candidates = payload.get("trials") or payload.get("measurements") or [payload]
        if not isinstance(candidates, list):
            candidates = [payload]
        for trial in candidates:
            if not isinstance(trial, dict):
                continue
            rows.append(
                {
                    "source": str(path),
                    "skill": str(trial.get("skill", trial.get("method", ""))),
                    "requested_distance_m": _number(
                        _first(trial, "requested_distance_m", "distance_m")
                    ),
                    "measured_displacement_m": _number(
                        _first(
                            trial,
                            "measured_displacement_m",
                            "settled_displacement_m",
                            "actual_distance_m",
                        )
                    ),
                    "stopping_drift_m": _number(
                        _first(trial, "stopping_drift_m", "braking_drift_m")
                    ),
                    "requested_angle_deg": _number(
                        _first(trial, "requested_angle_deg", "yaw_deg")
                    ),
                    "actual_angle_deg": _number(
                        _first(trial, "actual_angle_deg", "measured_yaw_deg")
                    ),
                }
            )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
