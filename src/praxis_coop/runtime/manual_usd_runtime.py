"""Manual USD runtime for the user's two-G1 long-box scene.

This is the first bridge from Praxis-Coop SkillGraphs to the hand-authored
`two_g1_box_push_layout.usda` scene. It uses USD/PXR directly, so it can run as
a fast layout-level adapter before we attach the same contract to live Isaac
Sim state and SONIC commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin
from pathlib import Path
import shutil

from pxr import Gf, Usd, UsdGeom

from praxis_coop.models import (
    ObjectState,
    Pose2D,
    RobotState,
    SceneGraph,
    SkillGraph,
    SkillResult,
    SkillStatus,
    TargetState,
    VerifierReport,
)
from praxis_coop.orchestrator import build_initial_skill_graph
from praxis_coop.skill_graph import execute_skill_graph
from praxis_coop.verifier import verify_execution
from praxis_coop.utils.geometry import apply_body_offset


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANUAL_SCENE = REPO_ROOT / "assets" / "manual_scenes" / "two_g1_box_push_layout.usda"


ROBOT_PATHS = {
    "g1": "/World/g1",
    "g1_01": "/World/g1_01",
}
OBJECT_PATH = "/World/long_box"
TARGET_PATH = "/World/target"
WALL_PATHS = {
    "wall_01": "/World/wall_01",
    "wall_03": "/World/wall_03",
    "wall_04": "/World/wall_04",
    "wall_05": "/World/wall_05",
    "wall_06": "/World/wall_06",
    "wall_07": "/World/wall_07",
}


@dataclass
class ManualUsdRuntimeResult:
    scene: SceneGraph
    skill_results: list[SkillResult]
    output_scene: Path
    metrics_path: Path | None = None


class ManualUsdSceneRuntime:
    """Read/write runtime for the manual USD layout."""

    def __init__(self, scene_path: str | Path = DEFAULT_MANUAL_SCENE, *, output_scene: str | Path | None = None):
        self.source_scene = Path(scene_path)
        self.output_scene = Path(output_scene) if output_scene is not None else self.source_scene
        if self.output_scene != self.source_scene:
            self.output_scene.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source_scene, self.output_scene)
        self.stage = _open_stage(self.output_scene)

    def read_scene_graph(self) -> SceneGraph:
        robots = {
            robot_id: RobotState(robot_id, self._pose2d(path))
            for robot_id, path in ROBOT_PATHS.items()
        }
        object_pose = self._pose2d(OBJECT_PATH)
        target_marker_pose = self._pose2d(TARGET_PATH)
        # The authored target marker is a location marker, not a payload yaw
        # command. Preserve the payload yaw from the physically valid initial
        # grasp/carry pose so the long box does not rotate through the robots.
        target_pose = Pose2D(target_marker_pose.x, target_marker_pose.y, object_pose.yaw)
        return SceneGraph(
            scene_id="manual_two_g1_box_push_layout_live_usd",
            robots=robots,
            movable_object=ObjectState(
                object_id="long_box",
                pose=object_pose,
                size=(2.2, 0.1, 0.1),
            ),
            gate=None,
            target=TargetState("target", target_pose, tolerance_xy=0.30, tolerance_yaw=0.18),
            obstacles=list(WALL_PATHS),
            metadata={
                "source_scene": str(self.source_scene),
                "output_scene": str(self.output_scene),
                "runtime": "manual_usd_xform_proxy",
                "note": "Layout-level USD xform adapter; not yet physics/contact simulation.",
            },
        )

    def execute_skill_graph(self, graph: SkillGraph) -> ManualUsdRuntimeResult:
        scene = self.read_scene_graph()
        adapter = _ManualUsdSkillAdapter(self, graph)
        skill_results = execute_skill_graph(graph, scene, adapter)
        self.save()
        return ManualUsdRuntimeResult(scene=scene, skill_results=skill_results, output_scene=self.output_scene)

    def save(self) -> None:
        self.stage.GetRootLayer().Save()

    def set_object_pose(self, pose: Pose2D) -> None:
        self._set_pose2d(OBJECT_PATH, pose, z=self._translation(OBJECT_PATH)[2])

    def set_robot_pose(self, robot_id: str, pose: Pose2D) -> None:
        self._set_pose2d(ROBOT_PATHS[robot_id], pose, z=self._translation(ROBOT_PATHS[robot_id])[2])

    def set_robots_from_object_offsets(self, object_pose: Pose2D, graph: SkillGraph) -> None:
        for robot_id, offset in graph.roles.offsets.items():
            self.set_robot_pose(robot_id, apply_body_offset(object_pose, offset))

    def _pose2d(self, path: str) -> Pose2D:
        x, y, _ = self._translation(path)
        yaw = self._yaw(path)
        return Pose2D(float(x), float(y), float(yaw))

    def _translation(self, path: str) -> tuple[float, float, float]:
        prim = _require_prim(self.stage, path)
        attr = prim.GetAttribute("xformOp:translate")
        if attr and attr.HasAuthoredValue():
            value = attr.Get()
            return (float(value[0]), float(value[1]), float(value[2]))
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = matrix.ExtractTranslation()
        return (float(t[0]), float(t[1]), float(t[2]))

    def _yaw(self, path: str) -> float:
        prim = _require_prim(self.stage, path)
        attr = prim.GetAttribute("xformOp:orient")
        if attr and attr.HasAuthoredValue():
            quat = attr.Get()
            w = float(quat.GetReal())
            x, y, z = [float(v) for v in quat.GetImaginary()]
            return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return 0.0

    def _set_pose2d(self, path: str, pose: Pose2D, *, z: float) -> None:
        prim = _require_prim(self.stage, path)
        _set_or_add_xform_op(prim, UsdGeom.XformOp.TypeTranslate, Gf.Vec3d(pose.x, pose.y, z))
        quat = Gf.Quatd(cos(pose.yaw / 2.0), Gf.Vec3d(0.0, 0.0, sin(pose.yaw / 2.0)))
        _set_or_add_xform_op(prim, UsdGeom.XformOp.TypeOrient, quat)


class _ManualUsdSkillAdapter:
    def __init__(self, runtime: ManualUsdSceneRuntime, graph: SkillGraph):
        self.runtime = runtime
        self.graph = graph

    def execute(self, node, scene: SceneGraph) -> SkillResult:
        if node.name == "approach_formation":
            self.runtime.set_robots_from_object_offsets(scene.movable_object.pose, self.graph)
            self._sync_scene_robot_poses(scene)
            return _skill_result(node, {"usd_xform_updates": 2.0})

        if node.name in {"establish_contact", "resync_contact"}:
            for robot in scene.robots.values():
                robot.in_contact = True
            return _skill_result(node, {"contact_count": float(len(scene.robots))})

        if node.name == "pre_align_object":
            target_yaw = scene.target.pose.yaw
            scene.movable_object.pose = Pose2D(scene.movable_object.pose.x, scene.movable_object.pose.y, target_yaw)
            self.runtime.set_object_pose(scene.movable_object.pose)
            self.runtime.set_robots_from_object_offsets(scene.movable_object.pose, self.graph)
            self._sync_scene_robot_poses(scene)
            return _skill_result(node, {"payload_yaw_rad": target_yaw})

        if node.name == "transport_object_path":
            max_speed = float(node.parameters.get("max_speed_mps", 0.15))
            if max_speed <= 0.12:
                next_pose = scene.target.pose
            else:
                next_pose = scene.target.pose.translated(-0.35, 0.08, 0.0)
            scene.movable_object.pose = next_pose
            scene.movable_object.speed = max_speed
            for robot in scene.robots.values():
                robot.in_contact = True
            self.runtime.set_object_pose(next_pose)
            self.runtime.set_robots_from_object_offsets(next_pose, self.graph)
            self._sync_scene_robot_poses(scene)
            return _skill_result(node, {"max_speed_mps": max_speed})

        if node.name == "release_and_exit":
            return _skill_result(node, {"released": 1.0})

        return _skill_result(node, {"usd_noop": 1.0})

    def _sync_scene_robot_poses(self, scene: SceneGraph) -> None:
        for robot_id in scene.robots:
            scene.robots[robot_id].pose = self.runtime._pose2d(ROBOT_PATHS[robot_id])


def _skill_result(node, metrics: dict[str, float]) -> SkillResult:
    phase_id = node.phase_id or (node.contact_flow.phase_id if node.contact_flow is not None else node.name)
    sync_group = node.sync_group or (node.contact_flow.sync_group if node.contact_flow is not None else "")
    return SkillResult(
        node.name,
        SkillStatus.SUCCESS,
        metrics,
        phase_id=phase_id,
        sync_group=sync_group,
    )


def run_manual_usd_skillgraph(
    instruction: str,
    *,
    scene_path: str | Path = DEFAULT_MANUAL_SCENE,
    output_scene: str | Path,
    transport_speed_mps: float | None = 0.10,
) -> tuple[ManualUsdRuntimeResult, VerifierReport]:
    runtime = ManualUsdSceneRuntime(scene_path, output_scene=output_scene)
    scene = runtime.read_scene_graph()
    orchestration = build_initial_skill_graph(instruction, scene)
    if transport_speed_mps is not None:
        for node in orchestration.graph.nodes:
            if node.name == "transport_object_path":
                node.parameters["max_speed_mps"] = transport_speed_mps
    result = runtime.execute_skill_graph(orchestration.graph)
    report = verify_execution(result.scene, result.skill_results)
    return result, report


def _open_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"Could not open USD scene: {path}")
    return stage


def _require_prim(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Missing prim: {path}")
    return prim


def _set_or_add_xform_op(prim, op_type, value) -> None:
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            op.Set(value)
            return
    xform.AddXformOp(op_type).Set(value)
