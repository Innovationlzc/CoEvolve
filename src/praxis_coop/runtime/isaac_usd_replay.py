"""USD replay writer for live Isaac Praxis-Coop rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from pathlib import Path
import shutil
from typing import Any


ROBOT_A_PATH = "/World/g1"
ROBOT_B_PATH = "/World/g1_01"
OBJECT_PATH = "/World/long_box"


@dataclass(frozen=True)
class UsdReplaySummary:
    output_scene: Path
    frame_count: int
    start_time_code: float
    end_time_code: float
    attempts: list[int]

    def to_json(self) -> dict[str, Any]:
        return {
            "output_scene": str(self.output_scene),
            "frame_count": self.frame_count,
            "start_time_code": self.start_time_code,
            "end_time_code": self.end_time_code,
            "attempts": self.attempts,
        }


def write_isaac_history_usd_replay(
    *,
    source_scene: str | Path,
    output_scene: str | Path,
    history: list[dict[str, float]],
    time_codes_per_second: float = 60.0,
    frame_stride: int = 2,
    attempt_gap_s: float = 1.0,
) -> UsdReplaySummary:
    """Write an animated USDA replay from live Isaac rollout history.

    The source scene is the user's hand-authored layout. We preserve its walls,
    target, robot payloads, and authored grasp geometry, then author time-sampled
    transforms for the two G1 roots and long box.
    """

    source = Path(source_scene)
    output = Path(output_scene)
    if not source.exists():
        raise FileNotFoundError(f"Missing source USD scene: {source}")
    if not history:
        raise ValueError("Cannot write USD replay without rollout history")

    from pxr import Gf, Usd, UsdGeom

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output), Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"Could not open replay USD scene for writing: {output}")

    fps = float(time_codes_per_second)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0.0)

    object_prim = _require_prim(stage, OBJECT_PATH)
    robot_a_prim = _require_prim(stage, ROBOT_A_PATH)
    robot_b_prim = _require_prim(stage, ROBOT_B_PATH)
    object_z = _translation(object_prim, Usd, UsdGeom)[2]
    robot_a_z = _translation(robot_a_prim, Usd, UsdGeom)[2]
    robot_b_z = _translation(robot_b_prim, Usd, UsdGeom)[2]
    object_yaw = _yaw(object_prim)
    robot_a_yaw = _yaw(robot_a_prim)
    robot_b_yaw = _yaw(robot_b_prim)

    object_translate = _xform_op(object_prim, UsdGeom, UsdGeom.XformOp.TypeTranslate)
    robot_a_translate = _xform_op(robot_a_prim, UsdGeom, UsdGeom.XformOp.TypeTranslate)
    robot_b_translate = _xform_op(robot_b_prim, UsdGeom, UsdGeom.XformOp.TypeTranslate)
    object_orient = _xform_op(object_prim, UsdGeom, UsdGeom.XformOp.TypeOrient)
    robot_a_orient = _xform_op(robot_a_prim, UsdGeom, UsdGeom.XformOp.TypeOrient)
    robot_b_orient = _xform_op(robot_b_prim, UsdGeom, UsdGeom.XformOp.TypeOrient)

    frame_count = 0
    end_time_code = 0.0
    attempts: list[int] = []
    current_attempt: int | None = None
    attempt_start_wall_t = 0.0
    timeline_offset_s = 0.0
    last_wall_t = 0.0

    rows = [row for index, row in enumerate(history) if index % max(1, frame_stride) == 0]
    if history[-1] is not rows[-1]:
        rows.append(history[-1])

    for row in rows:
        attempt = int(round(float(row.get("attempt", 1.0))))
        wall_t = float(row.get("wall_t", row.get("step", 0.0)))
        if current_attempt is None:
            current_attempt = attempt
            attempt_start_wall_t = wall_t
            attempts.append(attempt)
        elif attempt != current_attempt:
            timeline_offset_s += max(0.0, last_wall_t - attempt_start_wall_t) + attempt_gap_s
            current_attempt = attempt
            attempt_start_wall_t = wall_t
            attempts.append(attempt)

        replay_s = timeline_offset_s + max(0.0, wall_t - attempt_start_wall_t)
        time_code = round(replay_s * fps, 3)
        _set_translate(object_translate, Gf, Usd, row["mean_box_x"], row["mean_box_y"], object_z, time_code)
        _set_translate(robot_a_translate, Gf, Usd, row["mean_robot_a_x"], row["mean_robot_a_y"], robot_a_z, time_code)
        _set_translate(robot_b_translate, Gf, Usd, row["mean_robot_b_x"], row["mean_robot_b_y"], robot_b_z, time_code)
        _set_yaw(object_orient, Gf, Usd, object_yaw, time_code)
        _set_yaw(robot_a_orient, Gf, Usd, robot_a_yaw, time_code)
        _set_yaw(robot_b_orient, Gf, Usd, robot_b_yaw, time_code)
        end_time_code = max(end_time_code, time_code)
        last_wall_t = wall_t
        frame_count += 1

    stage.SetEndTimeCode(end_time_code)
    stage.GetRootLayer().Save()
    return UsdReplaySummary(
        output_scene=output,
        frame_count=frame_count,
        start_time_code=0.0,
        end_time_code=end_time_code,
        attempts=attempts,
    )


def _require_prim(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Missing prim in replay USD: {path}")
    return prim


def _translation(prim, Usd, UsdGeom) -> tuple[float, float, float]:
    attr = prim.GetAttribute("xformOp:translate")
    if attr and attr.HasAuthoredValue():
        value = attr.Get()
        return (float(value[0]), float(value[1]), float(value[2]))
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = matrix.ExtractTranslation()
    return (float(t[0]), float(t[1]), float(t[2]))


def _yaw(prim) -> float:
    attr = prim.GetAttribute("xformOp:orient")
    if not attr or not attr.HasAuthoredValue():
        return 0.0
    quat = attr.Get()
    w = float(quat.GetReal())
    x, y, z = [float(v) for v in quat.GetImaginary()]
    from math import atan2

    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _xform_op(prim, UsdGeom, op_type):
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    return xform.AddXformOp(op_type)


def _set_translate(op, Gf, Usd, x: float, y: float, z: float, time_code: float) -> None:
    op.Set(Gf.Vec3d(float(x), float(y), float(z)), Usd.TimeCode(time_code))


def _set_yaw(op, Gf, Usd, yaw: float, time_code: float) -> None:
    op.Set(Gf.Quatd(cos(yaw / 2.0), Gf.Vec3d(0.0, 0.0, sin(yaw / 2.0))), Usd.TimeCode(time_code))
