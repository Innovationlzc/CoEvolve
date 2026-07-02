"""Role allocation for two-humanoid transport."""

from __future__ import annotations

from .models import Pose2D, RoleAssignment, SceneGraph, TaskTaxonomy
from .utils.geometry import relative_body_offset


def allocate_roles(scene: SceneGraph, taxonomy: TaskTaxonomy) -> RoleAssignment:
    robot_ids = sorted(scene.robots)
    if len(robot_ids) < 2:
        raise ValueError("Praxis-Coop MVP requires at least two robots")

    if scene.scene_id.startswith("manual_two_g1_box_push_layout") or scene.scene_id == "stage1_two_g1_carry_wall":
        front = min(robot_ids, key=lambda rid: scene.robots[rid].pose.distance_to(scene.target.pose))
        back = [rid for rid in robot_ids if rid != front][0]
        return RoleAssignment(
            roles={
                front: "navigation_leader",
                back: "stability_follower",
            },
            formation="authored_payload_contact",
            offsets={
                robot_id: relative_body_offset(scene.movable_object.pose, scene.robots[robot_id].pose)
                for robot_id in robot_ids
            },
        )

    if scene.gate is not None:
        gate_x = scene.gate.center.x
        front = min(robot_ids, key=lambda rid: abs(scene.robots[rid].pose.x - gate_x))
        formation = "front_back"
        leader_offset = Pose2D(0.55, 0.0, 0.0)
        follower_offset = Pose2D(-0.55, 0.0, 0.0)
    else:
        front = min(robot_ids, key=lambda rid: scene.robots[rid].pose.distance_to(scene.target.pose))
        formation = "side_by_side_payload"
        leader_offset = Pose2D(0.0, 0.75, 0.0)
        follower_offset = Pose2D(0.0, -0.75, 0.0)
    back = [rid for rid in robot_ids if rid != front][0]

    return RoleAssignment(
        roles={
            front: "navigation_leader",
            back: "stability_follower",
        },
        formation=formation,
        offsets={
            front: leader_offset,
            back: follower_offset,
        },
    )
