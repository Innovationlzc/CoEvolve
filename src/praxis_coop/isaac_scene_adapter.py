"""SceneGraph adapters for the user's Isaac Lab cooperative transport scenes."""

from __future__ import annotations

from math import pi

from .models import GateState, ObjectState, Pose2D, RobotState, SceneGraph, TargetState


def manual_two_g1_box_push_scene_graph(*, at_target: bool = False) -> SceneGraph:
    """Return the user's hand-authored USD scene as a simulator-free graph.

    Source USD:
    `assets/manual_scenes/two_g1_box_push_layout.usda`.
    """

    object_pose = Pose2D(-1.50, 7.60, pi / 2.0) if at_target else Pose2D(-1.463371, 2.300997, pi / 2.0)
    return SceneGraph(
        scene_id="manual_two_g1_box_push_layout",
        robots={
            "g1": RobotState("g1", Pose2D(-1.75, 1.90, 0.0)),
            "g1_01": RobotState("g1_01", Pose2D(-1.75, 2.80, 0.0)),
        },
        movable_object=ObjectState(
            object_id="long_box",
            pose=object_pose,
            size=(2.2, 0.1, 0.1),
            speed=0.0,
        ),
        gate=None,
        target=TargetState(
            target_id="target",
            pose=Pose2D(-1.50, 7.60, pi / 2.0),
            tolerance_xy=0.30,
            tolerance_yaw=0.18,
        ),
        obstacles=["wall_01", "wall_03", "wall_04", "wall_05", "wall_06", "wall_07"],
        metadata={
            "source": "assets/manual_scenes/two_g1_box_push_layout.usda",
            "usd_kind": "hand_authored_layout",
            "robot_count": 2,
            "object_kind": "long_box",
            "primary_goal": "move long_box to target through wall corridor",
            "wall_poses": {
                "wall_01": (-0.15, 1.0, 0.35),
                "wall_03": (2.439544, 4.941329, 0.35),
                "wall_04": (1.005468, 4.914581, 0.35),
                "wall_05": (-0.829619, 3.492934, 0.35),
                "wall_06": (-0.15, 8.860252, 0.35),
                "wall_07": (-0.829619, 6.331826, 0.35),
            },
        },
    )


def stage1_two_g1_carry_wall_scene_graph(*, target_y: float = 3.75) -> SceneGraph:
    """Return the current Isaac Stage1 carry-place scene as a planning graph.

    This graph mirrors ``CoEvolve-TwoG1CarryPlace-Stage1Locomanip-v0``:
    two G1s start in the user-authored manual layout and the Stage1 objective
    is a short, straight +Y transport toward the first wall/corridor region.
    Robot ids intentionally match the IsaacLab scene keys used by the runtime.
    """

    object_pose = Pose2D(-1.4633708965114747, 2.300997045231125, pi / 2.0)
    return SceneGraph(
        scene_id="stage1_two_g1_carry_wall",
        robots={
            "robot": RobotState("robot", Pose2D(-1.75, 1.90, 0.0), role="stability_follower"),
            "robot_b": RobotState(
                "robot_b",
                Pose2D(-1.75, 2.80, 0.0),
                role="navigation_leader",
            ),
        },
        movable_object=ObjectState(
            object_id="long_box",
            pose=object_pose,
            size=(2.2, 0.1, 0.1),
            speed=0.0,
        ),
        gate=None,
        target=TargetState(
            target_id="stage1_wall_vicinity",
            pose=Pose2D(object_pose.x, target_y, object_pose.yaw),
            tolerance_xy=0.25,
            tolerance_yaw=0.18,
        ),
        obstacles=["wall_01", "wall_05"],
        metadata={
            "source": "CoEvolve-TwoG1CarryPlace-Stage1Locomanip-v0",
            "usd_source": "assets/manual_scenes/two_g1_box_push_layout.usda",
            "robot_count": 2,
            "object_kind": "long_box",
            "stage": "stage1_straight_wall_approach",
            "primary_goal": "stable synchronized short carry to first wall vicinity",
        },
    )


def figure2_passage_scene_graph(*, at_target: bool = False) -> SceneGraph:
    """Return the archived Figure-2 passage scene as a simulator-free graph.

    The values mirror `HaLyPO/coevolve/isaac_tasks/figure2_passage_carry`.
    Keeping them here lets the new method reason over the user's scene without
    importing Isaac during unit tests or LLM-side planning.
    """

    object_pose = Pose2D(3.05, 0.0, 0.0) if at_target else Pose2D(-0.10, 0.0, 0.08)
    return SceneGraph(
        scene_id="figure2_passage_carry",
        robots={
            "RobotA": RobotState("RobotA", Pose2D(-0.40, -0.82, 0.0), role="navigation_leader"),
            "RobotB": RobotState("RobotB", Pose2D(-0.40, 0.82, 0.0), role="stability_follower"),
        },
        movable_object=ObjectState(
            object_id="carton_20kg",
            pose=object_pose,
            size=(0.88, 0.55, 0.46),
            speed=0.0,
        ),
        gate=GateState(
            gate_id="narrow_gate",
            center=Pose2D(1.45, 0.0, 0.0),
            width=1.24,
            axis_yaw=0.0,
        ),
        target=TargetState(
            target_id="passage_goal",
            pose=Pose2D(3.05, 0.0, 0.0),
            tolerance_xy=0.20,
            tolerance_yaw=0.10,
        ),
        obstacles=[
            "PassageWallLeft",
            "PassageWallRight",
            "NarrowGateLeft",
            "NarrowGateRight",
        ],
        metadata={
            "source": "HaLyPO/coevolve/isaac_tasks/figure2_passage_carry",
            "robot_count": 2,
            "object_mass_kg": 20.0,
            "action_space": 11,
            "observation_space": 210,
            "controller_fidelity": "archived_contact_gated_tracker_proxy",
        },
    )
