"""Map cooperative SkillGraphs into SONIC / GEAR-SONIC planner commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, hypot, pi, sin
from typing import Any

from praxis_coop.models import Pose2D, SceneGraph, SkillGraph, SkillNode
from praxis_coop.sonic.commands import SonicCommand, SonicCommandStream, SonicLocomotionMode
from praxis_coop.sonic.cooperative_cfgen import CooperativeCFGen
from praxis_coop.utils.geometry import offset_path


@dataclass(frozen=True)
class SonicMappingConfig:
    """Conservative command-generation defaults for Phase 2."""

    approach_speed_mps: float = 0.22
    carry_speed_mps: float = 0.16
    primitive_speed_mps: float = 0.18
    turn_yaw_rate_radps: float = 0.45
    turn_segment_yaw_rad: float = pi / 4.0
    align_duration_s: float = 1.0
    contact_duration_s: float = 1.0
    hand_lock_duration_s: float = 0.4
    release_duration_s: float = 0.8
    min_segment_duration_s: float = 0.4
    idle_tick_s: float = 0.2


def map_skill_graph_to_sonic_stream(
    graph: SkillGraph,
    scene: SceneGraph,
    *,
    config: SonicMappingConfig | None = None,
) -> SonicCommandStream:
    """Convert a high-level cooperative graph into timestamped SONIC commands.

    The generated messages target GEAR-SONIC's planner interface. They are
    intentionally high-level direction/heading commands rather than joint or
    torque controls; SONIC remains responsible for balance and whole-body
    execution.
    """

    config = config or SonicMappingConfig()
    graph = CooperativeCFGen().annotate(graph, scene).graph
    stream = SonicCommandStream()
    robot_poses = {robot_id: robot.pose for robot_id, robot in scene.robots.items()}
    t = 0.0

    for node in graph.nodes:
        registry_skill = str(node.parameters.get("registry_skill", ""))
        if registry_skill in {"move_lateral", "move_longitudinal", "turn_body_yaw_360"}:
            commands, duration = _primitive_motion_commands(node, robot_poses, t, config)
        elif node.name == "approach_formation":
            commands, duration = _approach_commands(node, graph, scene, robot_poses, t, config)
        elif node.name == "establish_contact":
            commands, duration = _idle_all(
                node,
                robot_poses,
                t,
                config.contact_duration_s,
                phase="contact_hold",
                facing_pose=scene.movable_object.pose,
            )
        elif node.name == "lock_initial_hand_pose":
            commands, duration = _idle_all(
                node,
                robot_poses,
                t,
                config.hand_lock_duration_s,
                phase="hold_initial_payload_hand_pose",
                facing_pose=scene.movable_object.pose,
            )
        elif node.name == "pre_align_object":
            commands, duration = _idle_all(
                node,
                robot_poses,
                t,
                config.align_duration_s,
                phase="pre_align_hold",
                facing_pose=scene.target.pose,
            )
        elif node.name == "transport_object_path":
            commands, duration = _transport_commands(node, graph, scene, robot_poses, t, config)
        elif node.name == "release_and_exit":
            commands, duration = _idle_all(
                node,
                robot_poses,
                t,
                config.release_duration_s,
                phase="release_hold",
                facing_pose=scene.target.pose,
            )
        else:
            commands, duration = _idle_all(
                node,
                robot_poses,
                t,
                config.idle_tick_s,
                phase="unsupported_skill_idle",
                facing_pose=scene.target.pose,
            )

        stream.extend(commands)
        t += duration

    return stream.sorted()


def _approach_commands(
    node: SkillNode,
    graph: SkillGraph,
    scene: SceneGraph,
    robot_poses: dict[str, Pose2D],
    t: float,
    config: SonicMappingConfig,
) -> tuple[list[SonicCommand], float]:
    object_path = _object_path_from_node(graph, scene)
    contact_paths = _robot_paths_from_object_path(graph, object_path)
    commands: list[SonicCommand] = []
    durations = []

    for robot_id, current_pose in robot_poses.items():
        target_path = contact_paths.get(robot_id)
        if not target_path:
            continue
        target_pose = target_path[0]
        command = _segment_command(
            t=t,
            robot_id=robot_id,
            node=node,
            start=current_pose,
            target=target_pose,
            mode=SonicLocomotionMode.WALK,
            speed=config.approach_speed_mps,
            phase="walk_to_contact_pose",
            min_duration_s=config.min_segment_duration_s,
            facing_pose=scene.movable_object.pose,
        )
        commands.append(command)
        robot_poses[robot_id] = target_pose
        durations.append(command.duration_s)

    return commands, max(durations, default=config.idle_tick_s)


def _transport_commands(
    node: SkillNode,
    graph: SkillGraph,
    scene: SceneGraph,
    robot_poses: dict[str, Pose2D],
    t: float,
    config: SonicMappingConfig,
) -> tuple[list[SonicCommand], float]:
    object_path = _object_path_from_node(graph, scene)
    robot_paths = _robot_paths_from_object_path(graph, object_path)
    commands: list[SonicCommand] = []
    cursor = t

    if len(object_path) <= 1:
        return _idle_all(node, robot_poses, t, config.idle_tick_s, phase="empty_transport_path", facing_pose=scene.target.pose)

    max_speed = float(node.parameters.get("max_speed_mps", config.carry_speed_mps))
    speed = min(max_speed, config.carry_speed_mps)

    for seg_idx in range(1, len(object_path)):
        object_start = object_path[seg_idx - 1]
        object_target = object_path[seg_idx]
        segment_duration = max(
            _distance(object_start, object_target) / max(speed, 1.0e-6),
            config.min_segment_duration_s,
        )
        segment_commands: list[SonicCommand] = []
        for robot_id, path in robot_paths.items():
            if len(path) <= seg_idx:
                continue
            start = robot_poses.get(robot_id, path[seg_idx - 1])
            target = path[seg_idx]
            command = _segment_command(
                t=cursor,
                robot_id=robot_id,
                node=node,
                start=start,
                target=target,
                mode=SonicLocomotionMode.OBJECT_CARRYING,
                speed=speed,
                phase=f"carry_segment_{seg_idx}",
                phase_id=_phase_id(node, node.name, subphase=f"segment_{seg_idx}"),
                min_duration_s=segment_duration,
                facing_pose=object_target,
            )
            segment_commands.append(command)
            robot_poses[robot_id] = target
        commands.extend(segment_commands)
        cursor += segment_duration

    return commands, cursor - t


def _primitive_motion_commands(
    node: SkillNode,
    robot_poses: dict[str, Pose2D],
    t: float,
    config: SonicMappingConfig,
) -> tuple[list[SonicCommand], float]:
    skill = str(node.parameters.get("registry_skill", ""))
    if skill == "move_lateral":
        return _translation_primitive_commands(node, robot_poses, t, config, axis="lateral")
    if skill == "move_longitudinal":
        return _translation_primitive_commands(node, robot_poses, t, config, axis="longitudinal")
    if skill == "turn_body_yaw_360":
        return _turn_primitive_commands(node, robot_poses, t, config)
    raise ValueError(f"Unsupported primitive motion skill: {skill}")


def _translation_primitive_commands(
    node: SkillNode,
    robot_poses: dict[str, Pose2D],
    t: float,
    config: SonicMappingConfig,
    *,
    axis: str,
) -> tuple[list[SonicCommand], float]:
    distance = abs(float(node.parameters.get("distance_m", 0.0)))
    speed = min(float(node.parameters.get("max_speed_mps", config.primitive_speed_mps)), config.primitive_speed_mps)
    duration = max(distance / max(speed, 1.0e-6), config.min_segment_duration_s)
    reference_frame = str(node.parameters.get("reference_frame", "body"))
    direction = node.parameters.get("direction", "left" if axis == "lateral" else "forward")

    commands: list[SonicCommand] = []
    for robot_id, pose in sorted(robot_poses.items()):
        move_angle = _primitive_move_angle(pose, axis=axis, direction=direction, reference_frame=reference_frame)
        movement = (cos(move_angle), sin(move_angle), 0.0) if distance > 1.0e-6 else (0.0, 0.0, 0.0)
        target = Pose2D(
            pose.x + movement[0] * distance,
            pose.y + movement[1] * distance,
            pose.yaw,
        )
        commands.append(
            SonicCommand(
                t=t,
                robot_id=robot_id,
                skill_node=node.name,
                mode=SonicLocomotionMode.OBJECT_CARRYING
                if _hand_pose_policy(node) == "hold_initial_payload_hand_pose"
                else SonicLocomotionMode.WALK,
                movement=movement,
                facing=(cos(pose.yaw), sin(pose.yaw), 0.0),
                speed=speed,
                duration_s=duration,
                phase=f"{axis}_move_{_direction_label(direction)}",
                phase_id=_phase_id(node, node.name),
                start_xy=(pose.x, pose.y),
                target_xy=(target.x, target.y),
                hand_pose_policy=_hand_pose_policy(node),
                hand_reference=_hand_reference(node),
                motion_axis=axis,
                reference_frame=reference_frame,
                **_contact_flow_kwargs(node),
            )
        )
        robot_poses[robot_id] = target

    return commands, duration


def _turn_primitive_commands(
    node: SkillNode,
    robot_poses: dict[str, Pose2D],
    t: float,
    config: SonicMappingConfig,
) -> tuple[list[SonicCommand], float]:
    yaw_rate = min(
        abs(float(node.parameters.get("max_yaw_rate_radps", config.turn_yaw_rate_radps))),
        config.turn_yaw_rate_radps,
    )
    yaw_rate = max(yaw_rate, 1.0e-6)
    segment_yaw = abs(float(node.parameters.get("segment_yaw_rad", config.turn_segment_yaw_rad)))
    segment_yaw = min(max(segment_yaw, 1.0e-3), pi / 2.0)

    deltas = {robot_id: _turn_delta_for_pose(node, pose) for robot_id, pose in robot_poses.items()}
    max_delta = max((abs(delta) for delta in deltas.values()), default=0.0)
    segment_count = max(1, int(ceil(max_delta / segment_yaw)))
    segment_duration = max(segment_yaw / yaw_rate, config.min_segment_duration_s)

    commands: list[SonicCommand] = []
    for segment_idx in range(1, segment_count + 1):
        cursor = t + (segment_idx - 1) * segment_duration
        fraction = segment_idx / segment_count
        for robot_id, pose in sorted(robot_poses.items()):
            delta = deltas[robot_id]
            yaw = pose.yaw + delta * fraction
            commands.append(
                SonicCommand(
                    t=cursor,
                    robot_id=robot_id,
                    skill_node=node.name,
                    mode=SonicLocomotionMode.OBJECT_CARRYING
                    if _hand_pose_policy(node) == "hold_initial_payload_hand_pose"
                    else SonicLocomotionMode.WALK,
                    movement=(0.0, 0.0, 0.0),
                    facing=(cos(yaw), sin(yaw), 0.0),
                    speed=0.0,
                    duration_s=segment_duration,
                    phase=f"turn_yaw_segment_{segment_idx:02d}_of_{segment_count:02d}",
                    phase_id=_phase_id(node, node.name, subphase=f"turn_{segment_idx:02d}"),
                    start_xy=(pose.x, pose.y),
                    target_xy=(pose.x, pose.y),
                    hand_pose_policy=_hand_pose_policy(node),
                    hand_reference=_hand_reference(node),
                    motion_axis="yaw",
                    reference_frame=str(node.parameters.get("reference_frame", "body")),
                    delta_yaw_rad=delta,
                    target_yaw_rad=pose.yaw + delta,
                    **_contact_flow_kwargs(node),
                )
            )

    for robot_id, pose in list(robot_poses.items()):
        robot_poses[robot_id] = Pose2D(pose.x, pose.y, pose.yaw + deltas[robot_id])
    return commands, segment_count * segment_duration


def _primitive_move_angle(pose: Pose2D, *, axis: str, direction: Any, reference_frame: str) -> float:
    signed = _direction_sign(direction)
    if reference_frame == "world":
        base_angle = 0.0 if axis == "longitudinal" else pi / 2.0
    else:
        base_angle = pose.yaw if axis == "longitudinal" else pose.yaw + pi / 2.0
    return base_angle if signed >= 0.0 else base_angle + pi


def _direction_sign(direction: Any) -> float:
    if isinstance(direction, str):
        value = direction.strip().lower()
        if value in {"left", "forward", "positive", "+", "+1"}:
            return 1.0
        if value in {"right", "backward", "negative", "-", "-1"}:
            return -1.0
    numeric = float(direction)
    return 1.0 if numeric >= 0.0 else -1.0


def _direction_label(direction: Any) -> str:
    if isinstance(direction, str):
        return direction.strip().lower().replace(" ", "_")
    return "positive" if float(direction) >= 0.0 else "negative"


def _turn_delta_for_pose(node: SkillNode, pose: Pose2D) -> float:
    if "target_yaw_rad" in node.parameters:
        return float(node.parameters["target_yaw_rad"]) - pose.yaw
    if "delta_yaw_deg" in node.parameters:
        return float(node.parameters["delta_yaw_deg"]) * pi / 180.0
    delta = float(node.parameters.get("delta_yaw_rad", 0.0))
    return min(max(delta, -2.0 * pi), 2.0 * pi)


def _idle_all(
    node: SkillNode,
    robot_poses: dict[str, Pose2D],
    t: float,
    duration_s: float,
    *,
    phase: str,
    facing_pose: Pose2D,
) -> tuple[list[SonicCommand], float]:
    commands = [
        SonicCommand(
            t=t,
            robot_id=robot_id,
            skill_node=node.name,
            mode=SonicLocomotionMode.IDLE,
            movement=(0.0, 0.0, 0.0),
            facing=_facing_vector(pose, facing_pose),
            speed=0.0,
            duration_s=duration_s,
            phase=phase,
            phase_id=_phase_id(node, phase),
            start_xy=(pose.x, pose.y),
            target_xy=(pose.x, pose.y),
            hand_pose_policy=_hand_pose_policy(node),
            hand_reference=_hand_reference(node),
            **_contact_flow_kwargs(node),
        )
        for robot_id, pose in sorted(robot_poses.items())
    ]
    return commands, duration_s


def _segment_command(
    *,
    t: float,
    robot_id: str,
    node: SkillNode,
    start: Pose2D,
    target: Pose2D,
    mode: SonicLocomotionMode,
    speed: float,
    phase: str,
    min_duration_s: float,
    facing_pose: Pose2D,
    phase_id: str | None = None,
) -> SonicCommand:
    dx = target.x - start.x
    dy = target.y - start.y
    distance = hypot(dx, dy)
    if distance <= 1.0e-6:
        movement = (0.0, 0.0, 0.0)
    else:
        movement = (dx / distance, dy / distance, 0.0)
    duration_s = max(min_duration_s, distance / max(speed, 1.0e-6))
    return SonicCommand(
        t=t,
        robot_id=robot_id,
        skill_node=node.name,
        mode=mode,
        movement=movement,
        facing=_facing_vector(start, facing_pose),
        speed=speed,
        duration_s=duration_s,
        phase=phase,
        phase_id=phase_id or _phase_id(node, phase),
        start_xy=(start.x, start.y),
        target_xy=(target.x, target.y),
        hand_pose_policy=_hand_pose_policy(node),
        hand_reference=_hand_reference(node),
        **_contact_flow_kwargs(node),
    )


def _hand_pose_policy(node: SkillNode) -> str:
    if node.name == "lock_initial_hand_pose":
        return "hold_initial_payload_hand_pose"
    if node.parameters.get("upper_body_skill") == "hold_initial_payload_hand_pose":
        return "hold_initial_payload_hand_pose"
    return "none"


def _hand_reference(node: SkillNode) -> str:
    if _hand_pose_policy(node) == "hold_initial_payload_hand_pose":
        return str(node.parameters.get("reference_source") or "manual_scene_initial_pelvis_relative_pose")
    return ""


def _phase_id(node: SkillNode, fallback: str, *, subphase: str | None = None) -> str:
    base = node.phase_id
    if not base and node.contact_flow is not None:
        base = node.contact_flow.phase_id
    base = base or fallback
    return f"{base}.{subphase}" if subphase else base


def _contact_flow_kwargs(node: SkillNode) -> dict:
    spec = node.contact_flow
    contact = spec.contact if spec is not None else None
    object_goal = spec.object_goal if spec is not None else None
    sync_group = node.sync_group or (spec.sync_group if spec is not None else "")
    return {
        "object_goal_xy": (object_goal.x, object_goal.y) if object_goal is not None else None,
        "contact_intent": contact.mode if contact is not None else str(node.parameters.get("contact_intent", "none")),
        "contact_object_id": contact.object_id if contact is not None else str(node.parameters.get("object_id", "")),
        "contact_links": contact.links if contact is not None else tuple(node.parameters.get("contact_links", ())),
        "contact_target_region": contact.target_region if contact is not None else "",
        "sync_group": sync_group,
        "abort_on_contact_loss": bool(spec.abort_on_contact_loss if spec is not None else "contact_loss" in node.monitors),
        "allowed_contact_loss_s": float(spec.allowed_contact_loss_s if spec is not None else 0.0),
    }


def _object_path_from_node(graph: SkillGraph, scene: SceneGraph) -> list[Pose2D]:
    node = _find_node(graph, "transport_object_path")
    payload = node.parameters.get("object_path") or []
    path = [_pose_from_mapping(item) for item in payload if _is_pose_mapping(item)]
    if path:
        return path
    return [scene.movable_object.pose, scene.target.pose]


def _robot_paths_from_object_path(graph: SkillGraph, object_path: list[Pose2D]) -> dict[str, list[Pose2D]]:
    paths: dict[str, list[Pose2D]] = {}
    for robot_id, offset in graph.roles.offsets.items():
        paths[robot_id] = offset_path(object_path, offset)
    return paths


def _find_node(graph: SkillGraph, name: str) -> SkillNode:
    for node in graph.nodes:
        if node.name == name:
            return node
    raise ValueError(f"Skill node not found: {name}")


def _is_pose_mapping(value: Any) -> bool:
    return isinstance(value, dict) and "x" in value and "y" in value


def _pose_from_mapping(value: dict[str, Any]) -> Pose2D:
    return Pose2D(float(value["x"]), float(value["y"]), float(value.get("yaw", 0.0)))


def _facing_vector(start: Pose2D, target: Pose2D) -> tuple[float, float, float]:
    dx = target.x - start.x
    dy = target.y - start.y
    distance = hypot(dx, dy)
    if distance <= 1.0e-6:
        return (cos(start.yaw), sin(start.yaw), 0.0)
    return (dx / distance, dy / distance, 0.0)


def _distance(start: Pose2D, target: Pose2D) -> float:
    return hypot(target.x - start.x, target.y - start.y)
