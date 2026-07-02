"""Runtime adapters for skill execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fabs

from .models import SceneGraph, SkillNode, SkillResult, SkillStatus


@dataclass
class MockSonicAdapter:
    """Deterministic adapter used before real SONIC/ZMQ integration."""

    fail_nodes: set[str] = field(default_factory=set)

    def execute(self, node: SkillNode, scene: SceneGraph) -> SkillResult:
        if node.name in self.fail_nodes:
            return SkillResult(
                node_name=node.name,
                status=SkillStatus.FAILED,
                metrics={"simulated_failure": 1.0},
                message=f"Injected failure at {node.name}",
                phase_id=_node_phase_id(node),
                sync_group=_node_sync_group(node),
            )
        return SkillResult(
            node_name=node.name,
            status=SkillStatus.SUCCESS,
            metrics={"duration_s": min(node.timeout_s * 0.25, 5.0)},
            message="mock execution succeeded",
            phase_id=_node_phase_id(node),
            sync_group=_node_sync_group(node),
        )


@dataclass
class ProxyTransportAdapter:
    """Fast deterministic transport proxy for verifier-gated evolution tests.

    This adapter intentionally stays below Isaac/SONIC fidelity. Its job is to
    make SkillGraph patches executable in milliseconds before the same graph is
    sent to the real scene adapter.
    """

    stable_transport_speed_mps: float = 0.12
    strict_gate_yaw_rad: float = 0.06

    def execute(self, node: SkillNode, scene: SceneGraph) -> SkillResult:
        if node.name == "approach_formation":
            return _skill_result(node, SkillStatus.SUCCESS, {"duration_s": 2.0})

        if node.name in {"establish_contact", "resync_contact"}:
            for robot in scene.robots.values():
                robot.in_contact = True
            return _skill_result(node, SkillStatus.SUCCESS, {"contact_count": float(len(scene.robots))})

        if node.name == "pre_align_object":
            yaw_tolerance_deg = float(node.parameters.get("yaw_tolerance_deg", 7.0))
            if scene.gate is not None and yaw_tolerance_deg <= 5.0:
                scene.movable_object.pose = scene.movable_object.pose.__class__(
                    scene.movable_object.pose.x,
                    scene.movable_object.pose.y,
                    scene.gate.axis_yaw,
                )
            return _skill_result(node, SkillStatus.SUCCESS, {"yaw_tolerance_deg": yaw_tolerance_deg})

        if node.name == "transport_object_path":
            max_speed = float(node.parameters.get("max_speed_mps", 0.15))
            if scene.gate is not None:
                yaw_error = fabs(scene.movable_object.pose.yaw - scene.gate.axis_yaw)
            else:
                yaw_error = 0.0
            if max_speed <= self.stable_transport_speed_mps:
                scene.movable_object.pose = scene.target.pose
            else:
                scene.movable_object.pose = scene.target.pose.translated(-0.35, 0.08, 0.0)
            scene.movable_object.speed = max_speed
            return _skill_result(
                node,
                SkillStatus.SUCCESS,
                {"max_speed_mps": max_speed, "proxy_yaw_error_rad": yaw_error},
            )

        if node.name == "release_and_exit":
            return _skill_result(node, SkillStatus.SUCCESS, {"duration_s": 1.0})

        return _skill_result(node, SkillStatus.SUCCESS, {"duration_s": 0.5})


@dataclass
class SonicCommandEnvelope:
    """Command envelope for the future SONIC adapter."""

    robot_id: str
    command_type: str
    velocity_mps: float = 0.0
    heading_rad: float = 0.0
    body_height_m: float | None = None
    contact_intent: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def _skill_result(
    node: SkillNode,
    status: SkillStatus,
    metrics: dict[str, float],
    message: str = "",
) -> SkillResult:
    return SkillResult(
        node.name,
        status,
        metrics,
        message,
        phase_id=_node_phase_id(node),
        sync_group=_node_sync_group(node),
    )


def _node_phase_id(node: SkillNode) -> str:
    if node.phase_id:
        return node.phase_id
    if node.contact_flow is not None:
        return node.contact_flow.phase_id
    return node.name


def _node_sync_group(node: SkillNode) -> str:
    if node.sync_group:
        return node.sync_group
    if node.contact_flow is not None:
        return node.contact_flow.sync_group
    return ""
