"""Expose live release-gate evidence to planning without granting a dispatch.

The authoritative predicate remains in ``sonic.runtime_state``.  In particular,
an earlier eligible snapshot is not permission to release after a checkpoint
restore has moved the payload outside the physical support footprint.
"""

from __future__ import annotations

from math import hypot, isfinite
from typing import Any

from praxis_coop.sonic.runtime_state import (
    _payload_release_ready,
    payload_target_footprint_gap_m,
    wait_for_payload_release_ready,
)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def release_preflight_evidence(runtime_state: dict[str, Any]) -> dict[str, Any]:
    """Describe one full raw runtime snapshot using the *existing* release gate.

    This read-only evidence does not replace its freshness and continuous-time
    checks, does not choose an action, and does not change success tolerances.
    Incomplete/pruned telemetry cannot establish release eligibility.
    """

    report: dict[str, Any] = {
        "available": False,
        "eligible_at_snapshot": None,
        "blocked_reason": "complete raw runtime state is unavailable",
        "source": "live_runtime_snapshot",
        "dispatch_gate_still_required": True,
        "snapshot_stability_certified": False,
        "rule": (
            "Use this snapshot only as planning evidence. Re-check the live "
            "release gate after restore and immediately before dispatch. "
            "An earlier required release is not authorization at a changed "
            "state. Box/target footprint overlap alone does not establish "
            "COM support or successful placement."
        ),
    }
    if not isinstance(runtime_state, dict):
        return report
    payload = runtime_state.get("payload")
    target = runtime_state.get("target")
    robots = runtime_state.get("robots")
    control = runtime_state.get("control_health")
    if not all(isinstance(item, dict) for item in (payload, target, robots, control)):
        return report
    contacts = payload.get("contacts")
    if not isinstance(contacts, dict):
        return report
    required = {
        "payload.x": _number(payload.get("x")),
        "payload.y": _number(payload.get("y")),
        "payload.z": _number(payload.get("z")),
        "payload.tilt_deg": _number(payload.get("tilt_deg")),
        "target.center_x_m": _number(target.get("center_x_m")),
        "target.center_y_m": _number(target.get("center_y_m")),
        "target.radius_m": _number(target.get("radius_m")),
        "target.payload_center_target_z_m": _number(target.get("payload_center_target_z_m")),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        report["blocked_reason"] = "release geometry is missing or non-finite"
        report["missing_fields"] = missing
        return report

    # Read the dispatch API's defaults rather than introducing parallel limits.
    gate_defaults = wait_for_payload_release_ready.__kwdefaults__
    target_tolerance = gate_defaults["target_xy_tolerance_m"]
    max_drop = gate_defaults["max_drop_to_target_m"]
    eligible, reason = _payload_release_ready(
        runtime_state,
        target_xy_tolerance_m=target_tolerance,
        max_drop_to_target_m=max_drop,
    )
    dx = required["target.center_x_m"] - required["payload.x"]
    dy = required["target.center_y_m"] - required["payload.y"]
    xy_error = hypot(dx, dy)
    support_radius = max(0.0, required["target.radius_m"])
    drop = required["payload.z"] - required["target.payload_center_target_z_m"]
    footprint_gap = payload_target_footprint_gap_m(runtime_state)
    report.update({
        "available": True,
        "eligible_at_snapshot": eligible,
        "blocked_reason": "" if eligible else reason,
        "target_xy_error_m": xy_error,
        "target_support_radius_m": support_radius,
        "com_support_margin_m": support_radius - xy_error,
        "target_minus_payload_world_xy_m": [dx, dy],
        "target_footprint_gap_m": _number(footprint_gap),
        "payload_center_height_above_placed_center_m": drop,
        "payload_tilt_deg": required["payload.tilt_deg"],
        "payload_linear_speed_mps": _number(payload.get("linear_speed_mps")),
        "payload_angular_speed_radps": _number(payload.get("angular_speed_radps")),
        "gate_source": "sonic.runtime_state._payload_release_ready",
        "max_drop_to_target_m": max_drop,
        "captured_at_unix": _number(runtime_state.get("time_unix")),
        "state_sequence": runtime_state.get("state_sequence")
        if isinstance(runtime_state.get("state_sequence"), int) else None,
    })
    return report
