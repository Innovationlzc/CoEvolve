"""Skill graph generation and execution."""

from __future__ import annotations

from typing import Protocol

from .models import (
    ContactFlowSpec,
    ContactIntent,
    ExecutableSpec,
    Pose2D,
    RoleAssignment,
    SceneGraph,
    SkillGraph,
    SkillNode,
    SkillResult,
    SkillStatus,
    TaskTaxonomy,
    TransportPlan,
)


class SkillAdapter(Protocol):
    def execute(self, node: SkillNode, scene: SceneGraph) -> SkillResult:
        """Execute one skill node against a runtime or proxy."""


def generate_skill_graph(
    taxonomy: TaskTaxonomy,
    roles: RoleAssignment,
    specs: list[ExecutableSpec],
    transport_plan: TransportPlan,
) -> SkillGraph:
    leader = _robot_with_role(roles, "navigation_leader")
    follower = _robot_with_role(roles, "stability_follower")
    common_agents = {"leader": leader, "stabilizer": follower}
    spec_ids = [spec.spec_id for spec in specs]
    sync_group = "payload_team"
    payload_id = "payload"
    final_object_goal = transport_plan.object_path[-1] if transport_plan.object_path else None

    nodes = [
        SkillNode(
            name="approach_formation",
            agents=common_agents,
            parameters={
                "registry_skill": "walk_to_pose",
                "formation": roles.formation,
                "robot_waypoints": {
                    robot_id: len(path) for robot_id, path in transport_plan.robot_paths.items()
                },
            },
            preconditions=["scene_graph_available"],
            postconditions=["robots_at_contact_poses"],
            monitors=["robot_fall", "collision_pairs"],
            fallback=["stop_and_replan"],
            timeout_s=20,
            phase_id="approach_formation",
            sync_group=sync_group,
            contact_flow=_contact_flow(
                "approach_formation",
                0,
                payload_id,
                "none",
                sync_group=sync_group,
                target_region="formation_contact_pose",
                verifier_tags=("robot_fall", "collision_pairs"),
            ),
        ),
        SkillNode(
            name="establish_contact",
            agents=common_agents,
            parameters={"registry_skill": "establish_payload_contact", "contact_mode": "push_or_grasp"},
            preconditions=["robots_at_contact_poses"],
            postconditions=["both_agents_in_contact"],
            monitors=["contact_loss", "object_tilt"],
            fallback=["reapproach_contact"],
            timeout_s=15,
            phase_id="establish_contact",
            sync_group=sync_group,
            contact_flow=_contact_flow(
                "establish_contact",
                1,
                payload_id,
                "push_or_grasp",
                sync_group=sync_group,
                target_region="assigned_payload_face",
                maintain=True,
                abort_on_contact_loss=True,
                verifier_tags=("contact_loss", "object_tilt"),
            ),
        ),
        SkillNode(
            name="lock_initial_hand_pose",
            agents=common_agents,
            parameters={
                "registry_skill": "hold_initial_payload_hand_pose",
                "reference_source": "manual_scene_initial_state",
                "lock_frame": "initial_pelvis_relative_pose",
                "wrist_pose_tolerance_m": 0.08,
                "hand_joint_policy": "hold_initial",
                "maintain_during": ["pre_align_object", "transport_object_path"],
            },
            preconditions=["both_agents_in_contact"],
            postconditions=["initial_hand_pose_locked"],
            monitors=["wrist_drift", "contact_loss", "robot_fall"],
            fallback=["synchronized_stop", "relock_initial_hand_pose"],
            timeout_s=5,
            phase_id="lock_initial_hand_pose",
            sync_group=sync_group,
            contact_flow=_contact_flow(
                "lock_initial_hand_pose",
                2,
                payload_id,
                "hold_initial_payload_hand_pose",
                sync_group=sync_group,
                target_region="initial_pelvis_relative_pose",
                maintain=True,
                abort_on_contact_loss=True,
                verifier_tags=("wrist_drift", "contact_loss", "robot_fall"),
            ),
        ),
        SkillNode(
            name="pre_align_object",
            agents=common_agents,
            parameters={
                "registry_skill": "align_payload_to_corridor",
                "yaw_tolerance_deg": 7,
                "max_speed_mps": 0.12,
                "sync_predicates": transport_plan.sync_predicates,
                "upper_body_skill": "hold_initial_payload_hand_pose",
            },
            preconditions=["both_agents_in_contact", "initial_hand_pose_locked"],
            postconditions=["object_yaw_aligned_to_gate"],
            monitors=["clearance_violation", "object_tilt"],
            fallback=["stop_and_realign"],
            timeout_s=20,
            phase_id="pre_align_object",
            sync_group=sync_group,
            contact_flow=_contact_flow(
                "pre_align_object",
                3,
                payload_id,
                "maintain_payload_contact",
                sync_group=sync_group,
                object_goal=final_object_goal,
                target_region="corridor_entry_alignment",
                maintain=True,
                abort_on_contact_loss=True,
                verifier_tags=("clearance_violation", "object_tilt", "contact_loss"),
            ),
        ),
        SkillNode(
            name="transport_object_path",
            agents=common_agents,
            parameters={
                "registry_skill": "carry_payload_path",
                "object_path": [_pose_to_dict(pose) for pose in transport_plan.object_path],
                "object_path_waypoints": len(transport_plan.object_path),
                "path_strategy": "planned_object_waypoints",
                "robot_path_waypoints": {
                    robot_id: len(path) for robot_id, path in transport_plan.robot_paths.items()
                },
                "max_speed_mps": 0.15,
                "sync_predicates": transport_plan.sync_predicates,
                "upper_body_skill": "hold_initial_payload_hand_pose",
            },
            preconditions=["object_yaw_aligned_to_gate", "initial_hand_pose_locked"],
            postconditions=["object_at_target"],
            monitors=["collision_pairs", "contact_loss", "progress_stagnation", "wrist_drift"],
            fallback=["synchronized_stop", "reverse_to_safe_pose"],
            timeout_s=60,
            phase_id="transport_object_path",
            sync_group=sync_group,
            contact_flow=_contact_flow(
                "transport_object_path",
                4,
                payload_id,
                "carry_or_push_path",
                sync_group=sync_group,
                object_goal=final_object_goal,
                target_region="planned_object_path",
                maintain=True,
                abort_on_contact_loss=True,
                allowed_contact_loss_s=0.25,
                verifier_tags=("collision_pairs", "contact_loss", "progress_stagnation", "wrist_drift"),
            ),
        ),
        SkillNode(
            name="release_and_exit",
            agents=common_agents,
            parameters={"registry_skill": "release_payload"},
            preconditions=["object_at_target"],
            postconditions=["object_stable_and_robots_clear"],
            monitors=["object_tilt", "robot_fall"],
            fallback=["hold_position"],
            timeout_s=20,
            phase_id="release_and_exit",
            sync_group=sync_group,
            contact_flow=_contact_flow(
                "release_and_exit",
                5,
                payload_id,
                "release",
                links=(),
                sync_group=sync_group,
                object_goal=final_object_goal,
                target_region="target_zone",
                verifier_tags=("object_tilt", "robot_fall"),
            ),
        ),
    ]

    return SkillGraph(
        graph_id="two_g1_transport_v0",
        nodes=nodes,
        taxonomy=taxonomy,
        roles=roles,
        specs=spec_ids,
    )


def execute_skill_graph(
    graph: SkillGraph, scene: SceneGraph, adapter: SkillAdapter
) -> list[SkillResult]:
    results: list[SkillResult] = []
    for node in graph.nodes:
        result = adapter.execute(node, scene)
        results.append(result)
        if result.status != SkillStatus.SUCCESS:
            break
    return results


def _robot_with_role(roles: RoleAssignment, role: str) -> str:
    for robot_id, assigned in roles.roles.items():
        if assigned == role:
            return robot_id
    raise ValueError(f"Missing role assignment for {role}")


def _pose_to_dict(pose) -> dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "yaw": float(pose.yaw)}


def _contact_flow(
    phase_id: str,
    phase_index: int,
    object_id: str,
    mode: str,
    *,
    links: tuple[str, ...] = ("left_hand", "right_hand"),
    sync_group: str,
    object_goal: Pose2D | None = None,
    target_region: str = "",
    maintain: bool = False,
    abort_on_contact_loss: bool = False,
    allowed_contact_loss_s: float = 0.0,
    verifier_tags: tuple[str, ...] = (),
) -> ContactFlowSpec:
    return ContactFlowSpec(
        phase_id=phase_id,
        phase_index=phase_index,
        contact=ContactIntent(
            object_id=object_id,
            mode=mode,
            links=links,
            target_region=target_region,
            maintain=maintain,
        ),
        object_goal=object_goal,
        sync_group=sync_group,
        abort_on_contact_loss=abort_on_contact_loss,
        allowed_contact_loss_s=allowed_contact_loss_s,
        verifier_tags=verifier_tags,
    )
