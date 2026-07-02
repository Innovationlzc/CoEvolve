"""State and execution verifier."""

from __future__ import annotations

from math import fabs

from .models import (
    FailureMode,
    SceneGraph,
    SkillResult,
    SkillStatus,
    VerifierReport,
    wrap_angle,
)


def verify_execution(
    scene: SceneGraph,
    results: list[SkillResult],
    min_clearance_m: float = 0.12,
    yaw_tolerance_rad: float = 0.122,
) -> VerifierReport:
    metrics: dict[str, float] = {}
    violations: list[str] = []
    phase_trace = [
        {
            "node": result.node_name,
            "phase_id": result.phase_id or result.node_name,
            "status": result.status.value,
            "sync_group": result.sync_group,
        }
        for result in results
    ]

    failed = [result for result in results if result.status != SkillStatus.SUCCESS]
    if failed:
        failed_result = failed[0]
        return VerifierReport(
            success=False,
            failure_mode=FailureMode.ADAPTER_ERROR,
            metrics={**metrics, **failed_result.metrics},
            violations=[failed_result.message or failed_result.node_name],
            evidence={
                "failed_node": failed_result.node_name,
                "failed_phase_id": failed_result.phase_id or failed_result.node_name,
                "sync_group": failed_result.sync_group,
                "phase_trace": phase_trace,
                **failed_result.evidence,
            },
        )

    if scene.gate is not None:
        yaw_error = fabs(wrap_angle(scene.movable_object.pose.yaw - scene.gate.axis_yaw))
        clearance = (scene.gate.width - scene.movable_object.size[1]) / 2.0
        metrics["object_gate_yaw_error_rad"] = yaw_error
        metrics["estimated_gate_clearance_m"] = clearance
        if yaw_error > yaw_tolerance_rad:
            violations.append("object_yaw_not_aligned_to_gate")
        if clearance < min_clearance_m:
            violations.append("gate_clearance_too_small")

    target_error = scene.movable_object.pose.distance_to(scene.target.pose)
    metrics["target_error_m"] = target_error
    if target_error > scene.target.tolerance_xy:
        violations.append("object_not_at_target")

    contact_count = sum(1 for robot in scene.robots.values() if robot.in_contact)
    metrics["contact_count"] = float(contact_count)
    if contact_count < len(scene.robots):
        violations.append("not_all_agents_in_contact")

    if not violations:
        return VerifierReport(success=True, metrics=metrics, evidence={"phase_trace": phase_trace})

    if "gate_clearance_too_small" in violations:
        mode = FailureMode.CLEARANCE_VIOLATION
    elif "object_yaw_not_aligned_to_gate" in violations:
        mode = FailureMode.YAW_MISALIGNMENT
    elif "not_all_agents_in_contact" in violations:
        mode = FailureMode.CONTACT_LOSS
    else:
        mode = FailureMode.TARGET_MISS
    failed_phase = _phase_for_violation(violations, results)
    return VerifierReport(
        False,
        mode,
        metrics,
        violations,
        {
            "failed_phase_id": failed_phase,
            "phase_trace": phase_trace,
        },
    )


def _phase_for_violation(violations: list[str], results: list[SkillResult]) -> str:
    if not results:
        return ""
    if "not_all_agents_in_contact" in violations:
        for result in reversed(results):
            if result.node_name in {"establish_contact", "resync_contact", "transport_object_path"}:
                return result.phase_id or result.node_name
    if "object_not_at_target" in violations:
        for result in reversed(results):
            if result.node_name == "transport_object_path":
                return result.phase_id or result.node_name
    if "object_yaw_not_aligned_to_gate" in violations:
        for result in reversed(results):
            if result.node_name == "pre_align_object":
                return result.phase_id or result.node_name
    if "gate_clearance_too_small" in violations:
        for result in reversed(results):
            if result.node_name in {"pre_align_object", "transport_object_path"}:
                return result.phase_id or result.node_name
    last = results[-1]
    return last.phase_id or last.node_name
