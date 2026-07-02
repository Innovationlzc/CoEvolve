"""Live Isaac proxy runtime for the manual two-G1 box-push task.

This module intentionally avoids importing Isaac Lab at module import time. The
script that launches Isaac owns the AppLauncher lifecycle, then passes the live
Gym environment here for SkillGraph execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any

from praxis_coop.models import (
    ObjectState,
    Pose2D,
    RobotState,
    SceneGraph,
    SkillGraph,
    SkillResult,
    SkillStatus,
    TargetState,
)


DEFAULT_ISAAC_PROXY_TASK = "CoEvolve-TwoG1BoxPush-ManualLayout-Direct-v0"


@dataclass
class IsaacProxyExecutionConfig:
    """Execution parameters for mapping SkillGraph nodes to proxy actions."""

    max_steps: int = 420
    command_gain: float = 0.9
    stop_on_success: bool = True
    node_step_budgets: dict[str, int] = field(
        default_factory=lambda: {
            "approach_formation": 2,
            "establish_contact": 8,
            "pre_align_object": 2,
            "transport_object_path": 360,
            "release_and_exit": 8,
        }
    )


@dataclass
class IsaacProxyRuntimeResult:
    """Result of a SkillGraph rollout in the live Isaac proxy task."""

    scene: SceneGraph
    skill_results: list[SkillResult]
    history: list[dict[str, float]]
    first_success_step: int | None
    best_distance_m: float
    total_steps: int
    action_mode: str


def execute_isaac_proxy_skill_graph(
    env: Any,
    graph: SkillGraph,
    *,
    config: IsaacProxyExecutionConfig | None = None,
) -> IsaacProxyRuntimeResult:
    """Execute a SkillGraph by stepping a live IsaacLab Gym environment."""

    config = config or IsaacProxyExecutionConfig()
    torch = _torch()
    unwrapped = env.unwrapped
    action_mode = str(getattr(unwrapped.cfg, "action_mode", "unknown"))
    history: list[dict[str, float]] = []
    skill_results: list[SkillResult] = []
    total_steps = 0
    first_success_step: int | None = None
    best_distance = float("inf")
    contact_established = False
    terminal_success_scene: SceneGraph | None = None

    for node in graph.nodes:
        budget = int(config.node_step_budgets.get(node.name, 1))
        if node.name == "transport_object_path":
            budget = max(1, min(budget, config.max_steps - total_steps))
        node_start = total_steps
        node_best_distance = float("inf")
        node_success = True
        message = ""
        waypoints = _waypoints_for_node(node) if node.name == "transport_object_path" else []
        waypoint_idx = 0

        if node.name == "release_and_exit" and terminal_success_scene is not None:
            skill_results.append(
                SkillResult(
                    node.name,
                    SkillStatus.SUCCESS,
                    {
                        "steps": 0.0,
                        "best_distance_m": best_distance,
                        "total_steps": float(total_steps),
                    },
                    "release treated as logical success after terminal target reach",
                    phase_id=_node_phase_id(node),
                    sync_group=_node_sync_group(node),
                )
            )
            continue

        while total_steps < config.max_steps and total_steps - node_start < budget:
            target_xy = waypoints[waypoint_idx] if waypoints else None
            action = _action_for_node(env, node.name, config.command_gain, target_xy=target_xy)
            _, reward, terminated, truncated, _ = env.step(action)
            total_steps += 1

            snapshot = _snapshot(env, torch, reward, terminated, truncated)
            snapshot["step"] = float(total_steps)
            snapshot["node_index"] = float(len(skill_results))
            snapshot["node"] = float(_node_code(node.name))
            snapshot["phase"] = float(_node_code(_node_phase_id(node).split(".", 1)[0]))
            history.append(snapshot)

            best_distance = min(best_distance, snapshot["mean_distance"])
            node_best_distance = min(node_best_distance, snapshot["mean_distance"])
            if first_success_step is None and snapshot["success_event_rate"] > 0.0:
                first_success_step = total_steps
                terminal_success_scene = _scene_from_env(env, torch, contact_established=True)

            if node.name == "establish_contact":
                contact_established = True

            if node.name == "transport_object_path":
                if target_xy is not None and _distance_to_xy(env, torch, target_xy) <= float(unwrapped.cfg.success_tolerance):
                    waypoint_idx = min(waypoint_idx + 1, len(waypoints) - 1)
                if snapshot["mean_distance"] <= float(unwrapped.cfg.success_tolerance):
                    break
                if config.stop_on_success and snapshot["success_event_rate"] > 0.0:
                    break

            if bool((terminated | truncated).detach().any().cpu().item()):
                if config.stop_on_success and snapshot["success_event_rate"] > 0.0:
                    break

        if node.name == "transport_object_path" and node_best_distance > float(unwrapped.cfg.success_tolerance):
            node_success = False
            message = "transport_object_path did not reach target within live Isaac proxy budget"

        status = SkillStatus.SUCCESS if node_success else SkillStatus.FAILED
        skill_results.append(
            SkillResult(
                node.name,
                status,
                {
                    "steps": float(total_steps - node_start),
                    "best_distance_m": node_best_distance,
                    "total_steps": float(total_steps),
                },
                message,
                phase_id=_node_phase_id(node),
                sync_group=_node_sync_group(node),
            )
        )
        if status != SkillStatus.SUCCESS:
            break

    if first_success_step is not None:
        contact_established = True
    final_scene = terminal_success_scene or _scene_from_env(env, torch, contact_established=contact_established)
    return IsaacProxyRuntimeResult(
        scene=final_scene,
        skill_results=skill_results,
        history=history,
        first_success_step=first_success_step,
        best_distance_m=best_distance,
        total_steps=total_steps,
        action_mode=action_mode,
    )


def _action_for_node(env: Any, node_name: str, command_gain: float, *, target_xy: tuple[float, float] | None = None):
    torch = _torch()
    if node_name in {"approach_formation", "pre_align_object", "release_and_exit"}:
        return torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    return _goal_directed_action(env, command_gain, target_xy=target_xy)


def _goal_directed_action(env: Any, command_gain: float, *, target_xy: tuple[float, float] | None = None):
    torch = _torch()
    unwrapped = env.unwrapped
    box_pos = unwrapped.box.data.root_pos_w.torch - unwrapped.scene.env_origins
    if target_xy is None:
        goal_xy = unwrapped.goal_pos[:, 0:2]
    else:
        goal_xy = torch.tensor(target_xy, dtype=box_pos.dtype, device=unwrapped.device).repeat(unwrapped.num_envs, 1)
    delta = goal_xy - box_pos[:, 0:2]
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp(min=1.0e-6)
    command = (delta / distance) * float(command_gain)
    action = torch.zeros(env.action_space.shape, device=unwrapped.device)
    action[:, 0:2] = command
    action[:, 2:4] = command
    return action.clamp(-1.0, 1.0)


def _waypoints_for_node(node: Any) -> list[tuple[float, float]]:
    payload = node.parameters.get("object_path") or []
    waypoints: list[tuple[float, float]] = []
    for item in payload:
        try:
            waypoints.append((float(item["x"]), float(item["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(waypoints) <= 1:
        return []
    return waypoints[1:]


def _distance_to_xy(env: Any, torch: Any, target_xy: tuple[float, float]) -> float:
    unwrapped = env.unwrapped
    box_pos = unwrapped.box.data.root_pos_w.torch - unwrapped.scene.env_origins
    target = torch.tensor(target_xy, dtype=box_pos.dtype, device=unwrapped.device).repeat(unwrapped.num_envs, 1)
    distance = torch.linalg.norm(target - box_pos[:, 0:2], dim=-1)
    return _mean_float(distance)


def _snapshot(env: Any, torch: Any, reward: Any, terminated: Any, truncated: Any) -> dict[str, float]:
    unwrapped = env.unwrapped
    box_pos = unwrapped.box.data.root_pos_w.torch - unwrapped.scene.env_origins
    goal_xy = unwrapped.goal_pos[:, 0:2]
    distance = torch.linalg.norm(goal_xy - box_pos[:, 0:2], dim=-1)
    success_event_rate = _mean_float(unwrapped.last_success)
    done_rate = _mean_float(terminated | truncated)
    use_terminal_state = success_event_rate > 0.0 or done_rate > 0.0
    metric_box_pos = unwrapped.last_box_pos if use_terminal_state else box_pos
    metric_distance = unwrapped.last_distance if use_terminal_state else distance
    robot_a_pos = unwrapped.robot_a.data.root_pos_w.torch[:, :3] - unwrapped.scene.env_origins
    robot_b_pos = unwrapped.robot_b.data.root_pos_w.torch[:, :3] - unwrapped.scene.env_origins
    metric_robot_a_pos = unwrapped.last_robot_a_pos if use_terminal_state else robot_a_pos
    metric_robot_b_pos = unwrapped.last_robot_b_pos if use_terminal_state else robot_b_pos
    return {
        "mean_reward": _mean_float(reward),
        "mean_box_x": _mean_float(metric_box_pos[:, 0]),
        "mean_box_y": _mean_float(metric_box_pos[:, 1]),
        "mean_box_z": _mean_float(metric_box_pos[:, 2]),
        "mean_robot_a_x": _mean_float(metric_robot_a_pos[:, 0]),
        "mean_robot_a_y": _mean_float(metric_robot_a_pos[:, 1]),
        "mean_robot_a_z": _mean_float(metric_robot_a_pos[:, 2]),
        "mean_robot_b_x": _mean_float(metric_robot_b_pos[:, 0]),
        "mean_robot_b_y": _mean_float(metric_robot_b_pos[:, 1]),
        "mean_robot_b_z": _mean_float(metric_robot_b_pos[:, 2]),
        "mean_distance": _mean_float(metric_distance),
        "success_rate": _mean_float(metric_distance < unwrapped.cfg.success_tolerance),
        "success_event_rate": success_event_rate,
        "done_rate": done_rate,
    }


def _scene_from_env(env: Any, torch: Any, *, contact_established: bool) -> SceneGraph:
    unwrapped = env.unwrapped
    box_pos = unwrapped.box.data.root_pos_w.torch - unwrapped.scene.env_origins
    robot_a_pos = unwrapped.robot_a.data.root_pos_w.torch[:, :3] - unwrapped.scene.env_origins
    robot_b_pos = unwrapped.robot_b.data.root_pos_w.torch[:, :3] - unwrapped.scene.env_origins
    if _mean_float(unwrapped.last_success) > 0.0:
        box_pos = unwrapped.last_box_pos
        robot_a_pos = unwrapped.last_robot_a_pos
        robot_b_pos = unwrapped.last_robot_b_pos
    box_xy = _first_xy(box_pos, torch)
    robot_a_xy = _first_xy(robot_a_pos, torch)
    robot_b_xy = _first_xy(robot_b_pos, torch)
    goal_xy = _first_xy(unwrapped.goal_pos, torch)
    box_yaw = float(getattr(unwrapped.cfg, "box_start_yaw", pi / 2.0))
    return SceneGraph(
        scene_id="manual_two_g1_box_push_live_isaac_proxy",
        robots={
            "g1": RobotState("g1", Pose2D(robot_a_xy[0], robot_a_xy[1], 0.0), in_contact=contact_established),
            "g1_01": RobotState("g1_01", Pose2D(robot_b_xy[0], robot_b_xy[1], 0.0), in_contact=contact_established),
        },
        movable_object=ObjectState(
            "long_box",
            Pose2D(box_xy[0], box_xy[1], box_yaw),
            size=(2.2, 0.1, 0.1),
        ),
        gate=None,
        target=TargetState(
            "target",
            Pose2D(goal_xy[0], goal_xy[1], box_yaw),
            tolerance_xy=float(unwrapped.cfg.success_tolerance),
            tolerance_yaw=0.18,
        ),
        obstacles=["wall_01", "wall_03", "wall_04", "wall_05", "wall_06", "wall_07"],
        metadata={
            "runtime": "live_isaac_proxy",
            "task": DEFAULT_ISAAC_PROXY_TASK,
            "action_mode": str(getattr(unwrapped.cfg, "action_mode", "unknown")),
        },
    )


def _first_xy(tensor: Any, torch: Any) -> tuple[float, float]:
    value = tensor.detach().float()
    if value.ndim == 2:
        value = value[0]
    value = value.cpu()
    return (float(value[0].item()), float(value[1].item()))


def _mean_float(value: Any) -> float:
    return float(value.detach().float().mean().cpu().item())


def _node_code(node_name: str) -> int:
    codes = {
        "approach_formation": 1,
        "establish_contact": 2,
        "pre_align_object": 3,
        "transport_object_path": 4,
        "release_and_exit": 5,
    }
    return codes.get(node_name, 0)


def _node_phase_id(node) -> str:
    if getattr(node, "phase_id", ""):
        return node.phase_id
    if getattr(node, "contact_flow", None) is not None:
        return node.contact_flow.phase_id
    return node.name


def _node_sync_group(node) -> str:
    if getattr(node, "sync_group", ""):
        return node.sync_group
    if getattr(node, "contact_flow", None) is not None:
        return node.contact_flow.sync_group
    return ""


def _torch():
    import torch

    return torch
