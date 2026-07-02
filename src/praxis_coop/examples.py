"""Small deterministic scenes for tests and demos."""

from __future__ import annotations

from .models import GateState, ObjectState, Pose2D, RobotState, SceneGraph, TargetState


def two_g1_box_scene(object_at_target: bool = True) -> SceneGraph:
    object_pose = Pose2D(3.0, 0.0, 0.0) if object_at_target else Pose2D(0.0, 0.0, 0.0)
    return SceneGraph(
        scene_id="two_g1_box_gate_proxy",
        robots={
            "G1-front": RobotState("G1-front", Pose2D(0.6, 0.0, 0.0), in_contact=True),
            "G1-back": RobotState("G1-back", Pose2D(-0.6, 0.0, 0.0), in_contact=True),
        },
        movable_object=ObjectState(
            object_id="box_01",
            pose=object_pose,
            size=(0.8, 0.45, 0.35),
        ),
        gate=GateState(
            gate_id="gate_01",
            center=Pose2D(1.5, 0.0, 0.0),
            width=0.9,
            axis_yaw=0.0,
        ),
        target=TargetState("target_zone", Pose2D(3.0, 0.0, 0.0)),
        obstacles=["left_wall", "right_wall"],
    )
