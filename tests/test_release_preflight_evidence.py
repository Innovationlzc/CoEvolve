"""Offline regressions for planning/execution release-boundary drift."""

from copy import deepcopy
import json

import pytest

from praxis_coop.cap_policy.release_preflight_evidence import release_preflight_evidence
from praxis_coop.sonic.runtime_state import _payload_release_ready


def _state():
    return {
        "payload": {
            "active": True,
            "x": 0.25, "y": 0.0, "z": 0.85,
            "tilt_deg": 0.0, "yaw_rad": 0.0,
            "half_extents_xyz_m": [0.9, 0.05, 0.05],
            "linear_speed_mps": 0.0, "angular_speed_radps": 0.0,
            "contacts": {"support_robots": ["front", "back"], "floor": False, "walls": []},
            "support_polygon": {"available": True, "inside": True},
        },
        "target": {"available": True, "center_x_m": 0.0, "center_y_m": 0.0,
                   "radius_m": 0.3, "payload_center_target_z_m": 0.65},
        "control_health": {"healthy": True},
        "robots": {"front": {"pelvis_height_m": 0.78}, "back": {"pelvis_height_m": 0.78}},
    }


def test_snapshot_eligibility_does_not_authorize_release_after_restore_drift():
    state = _state()
    state["payload"]["x"] = 0.29995380964446056
    before = release_preflight_evidence(state)
    assert before["eligible_at_snapshot"] is True
    state["payload"]["x"] = 0.305
    after = release_preflight_evidence(state)
    assert after["eligible_at_snapshot"] is False
    assert after["com_support_margin_m"] == pytest.approx(-0.005)
    assert "COM projection is outside" in after["blocked_reason"]
    assert before["dispatch_gate_still_required"] is True
    assert after["snapshot_stability_certified"] is False


def test_long_box_overlap_is_not_com_support():
    state = _state()
    state["payload"]["x"] = 0.305
    report = release_preflight_evidence(state)
    assert report["target_footprint_gap_m"] == 0.0
    assert report["eligible_at_snapshot"] is False


def test_recorded_seed201_posture_needs_existing_pre_lower_gate():
    state = _state()
    state["payload"].update(x=0.29995380964446056, z=1.0958493969246539, tilt_deg=32.41411671029502)
    report = release_preflight_evidence(state)
    assert report["eligible_at_snapshot"] is False
    assert "level or pre-lowered" in report["blocked_reason"]
    state["payload"]["z"] = 0.86
    assert release_preflight_evidence(state)["eligible_at_snapshot"] is True


@pytest.mark.parametrize("change", [
    {"payload": {"active": False}},
    {"payload": {"tilt_deg": 95.0}},
    {"payload": {"z": 1.3}},
    {"payload": {"support_polygon": {"available": True, "inside": False}}},
    {"payload": {"contacts": {"support_robots": ["front"], "floor": False, "walls": []}}},
    {"payload": {"contacts": {"support_robots": ["front", "back"], "floor": True, "walls": []}}},
    {"control_health": {"healthy": False}},
])
def test_report_uses_authoritative_predicate_without_relaxing_checks(change):
    state = _state()
    for key, fields in change.items():
        state[key].update(fields)
    expected, reason = _payload_release_ready(state, target_xy_tolerance_m=None, max_drop_to_target_m=0.45)
    report = release_preflight_evidence(state)
    assert report["eligible_at_snapshot"] is expected is False
    assert report["blocked_reason"] == reason


def test_evidence_is_read_only_and_json_serializable():
    state = _state()
    saved = deepcopy(state)
    report = release_preflight_evidence(state)
    assert state == saved
    json.dumps(report, allow_nan=False)
    assert report["target_minus_payload_world_xy_m"] == [-0.25, 0.0]


@pytest.mark.parametrize("state", [None, {}, {"payload": {"active": True}, "target": {"available": True}}])
def test_pruned_or_missing_telemetry_is_not_reported_as_eligible(state):
    report = release_preflight_evidence(state)
    assert report["available"] is False
    assert report["eligible_at_snapshot"] is None


def test_nonfinite_geometry_is_not_reported_as_eligible():
    state = _state()
    state["payload"]["x"] = float("nan")
    report = release_preflight_evidence(state)
    assert report["available"] is False
    assert "payload.x" in report["missing_fields"]
    json.dumps(report, allow_nan=False)
