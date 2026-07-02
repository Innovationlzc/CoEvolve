"""Live Isaac runtime driven by SONIC / GEAR-SONIC command streams.

This is the first version of the missing middle layer. It consumes the same
planner command stream that can be sent to GEAR-SONIC over ZMQ, then adapts the
per-robot movement intents to the current IsaacLab manual-layout task action
space. The backend is still the task's 4D command interface; it is not yet
joint-level SONIC whole-body control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
import time
from typing import Any

from praxis_coop.models import (
    ObjectState,
    Pose2D,
    RobotState,
    SceneGraph,
    SkillResult,
    SkillStatus,
    TargetState,
)
from praxis_coop.sonic.commands import SonicCommand, SonicCommandStream, SonicLocomotionMode


@dataclass
class IsaacSonicExecutionConfig:
    """Execution parameters for replaying SONIC command streams in Isaac."""

    max_steps: int = 620
    command_gain: float = 0.9
    command_time_scale: float = 0.35
    sleep_per_step_s: float = 0.0
    stop_on_success: bool = True
    contact_skill_names: set[str] = field(
        default_factory=lambda: {
            "establish_contact",
            "pre_align_object",
            "transport_object_path",
            "release_and_exit",
        }
    )


@dataclass
class IsaacSonicRuntimeResult:
    """Result of one Isaac rollout driven by a SONIC command stream."""

    scene: SceneGraph
    skill_results: list[SkillResult]
    history: list[dict[str, float]]
    first_success_step: int | None
    best_distance_m: float
    total_steps: int
    action_mode: str
    stream_duration_s: float


def execute_isaac_sonic_command_stream(
    env: Any,
    stream: SonicCommandStream,
    *,
    config: IsaacSonicExecutionConfig | None = None,
) -> IsaacSonicRuntimeResult:
    """Step a live IsaacLab environment from timestamped SONIC commands."""

    config = config or IsaacSonicExecutionConfig()
    torch = _torch()
    unwrapped = env.unwrapped
    action_mode = str(getattr(unwrapped.cfg, "action_mode", "unknown"))
    by_robot = stream.by_robot()
    robot_ids = _robot_ids(by_robot)
    sim_dt = float(unwrapped.cfg.sim.dt * unwrapped.cfg.decimation)
    max_steps = min(config.max_steps, int(getattr(unwrapped, "max_episode_length", config.max_steps)))

    history: list[dict[str, float]] = []
    first_success_step: int | None = None
    best_distance = float("inf")
    total_steps = 0
    terminal_success_scene: SceneGraph | None = None
    contact_established = False

    while total_steps < max_steps:
        t_wall = total_steps * sim_dt
        t_stream = t_wall / max(config.command_time_scale, 1.0e-6)
        active = {
            robot_id: _active_command(commands, t_stream)
            for robot_id, commands in by_robot.items()
        }
        if any(command and command.skill_node in config.contact_skill_names for command in active.values()):
            contact_established = True

        action = _action_from_active_commands(env, active, robot_ids, config.command_gain)
        _, reward, terminated, truncated, _ = env.step(action)
        if config.sleep_per_step_s > 0.0:
            time.sleep(config.sleep_per_step_s)
        total_steps += 1

        snapshot = _snapshot(env, torch, reward, terminated, truncated)
        snapshot["step"] = float(total_steps)
        snapshot["stream_t"] = float(t_stream)
        snapshot["wall_t"] = float(t_wall)
        snapshot["active_mode_g1"] = float(_mode_code(active.get("g1")))
        snapshot["active_mode_g1_01"] = float(_mode_code(active.get("g1_01")))
        snapshot["active_skill"] = float(_skill_code(_first_active_skill(active)))
        snapshot["active_phase"] = float(_phase_code(_first_active_phase(active)))
        history.append(snapshot)

        best_distance = min(best_distance, snapshot["mean_distance"])
        if first_success_step is None and snapshot["success_event_rate"] > 0.0:
            first_success_step = total_steps
            terminal_success_scene = _scene_from_env(env, torch, contact_established=True)

        done = bool((terminated | truncated).detach().any().cpu().item())
        if config.stop_on_success and snapshot["success_event_rate"] > 0.0:
            break
        if done:
            break
        if t_stream > stream.duration_s + 0.5 and snapshot["mean_distance"] <= float(unwrapped.cfg.success_tolerance):
            break

    final_scene = terminal_success_scene or _scene_from_env(env, torch, contact_established=contact_established)
    skill_results = _skill_results_from_rollout(env, stream, first_success_step, best_distance, total_steps)
    return IsaacSonicRuntimeResult(
        scene=final_scene,
        skill_results=skill_results,
        history=history,
        first_success_step=first_success_step,
        best_distance_m=best_distance,
        total_steps=total_steps,
        action_mode=action_mode,
        stream_duration_s=stream.duration_s,
    )


def _action_from_active_commands(
    env: Any,
    active: dict[str, SonicCommand | None],
    robot_ids: list[str],
    command_gain: float,
):
    torch = _torch()
    action_mode = str(getattr(env.unwrapped.cfg, "action_mode", "unknown"))
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    for slot, robot_id in enumerate(robot_ids[:2]):
        command = active.get(robot_id)
        if command is None or command.mode == SonicLocomotionMode.IDLE:
            vec = (0.0, 0.0)
        else:
            # In proxy_box_velocity, the env action is a normalized direction,
            # not a physical m/s target. Keep SONIC speed in the command log/ZMQ
            # envelope, but do not double-scale the normalized action here.
            if action_mode in {"proxy_box_velocity", "base_command"}:
                speed_scale = 1.0
            else:
                speed_scale = max(0.15, min(1.0, float(command.speed) / 0.25)) if command.speed > 0 else 1.0
            vec = (
                float(command.movement[0]) * float(command_gain) * speed_scale,
                float(command.movement[1]) * float(command_gain) * speed_scale,
            )
        action[:, slot * 2 : slot * 2 + 2] = torch.tensor(vec, dtype=action.dtype, device=action.device)
    return action.clamp(-1.0, 1.0)


def _active_command(commands: list[SonicCommand], t_stream: float) -> SonicCommand | None:
    if not commands:
        return None
    latest: SonicCommand | None = None
    for command in commands:
        if command.t <= t_stream <= command.t + command.duration_s:
            return command
        if command.t <= t_stream:
            latest = command
    return latest if latest and t_stream <= latest.t + latest.duration_s + 0.5 else None


def _robot_ids(by_robot: dict[str, list[SonicCommand]]) -> list[str]:
    ids = sorted(by_robot)
    for expected in ("g1", "g1_01"):
        if expected not in ids:
            ids.append(expected)
    return ids[:2]


def _skill_results_from_rollout(
    env: Any,
    stream: SonicCommandStream,
    first_success_step: int | None,
    best_distance_m: float,
    total_steps: int,
) -> list[SkillResult]:
    unwrapped = env.unwrapped
    node_names = []
    phase_by_node = {}
    sync_by_node = {}
    for command in stream.sorted().commands:
        if command.skill_node not in node_names:
            node_names.append(command.skill_node)
            phase_by_node[command.skill_node] = command.phase_id or command.phase
            sync_by_node[command.skill_node] = command.sync_group
    if not node_names:
        node_names = ["sonic_command_stream"]

    reached = best_distance_m <= float(unwrapped.cfg.success_tolerance) or first_success_step is not None
    results: list[SkillResult] = []
    for node_name in node_names:
        if reached:
            status = SkillStatus.SUCCESS
            message = "SONIC command stream reached target in Isaac adapter"
        elif node_name == "transport_object_path":
            status = SkillStatus.FAILED
            message = "SONIC command stream did not drive the payload to target within Isaac budget"
        else:
            status = SkillStatus.SUCCESS
            message = "logical skill phase replayed"
        results.append(
            SkillResult(
                node_name,
                status,
                {
                    "best_distance_m": float(best_distance_m),
                    "total_steps": float(total_steps),
                },
                message,
                phase_id=phase_by_node.get(node_name, node_name),
                sync_group=sync_by_node.get(node_name, ""),
            )
        )
        if status == SkillStatus.FAILED:
            break
    return results


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
    if _mean_float(unwrapped.last_success) > 0.0 or _mean_float(unwrapped.last_distance) < float("inf"):
        box_pos = unwrapped.last_box_pos
        robot_a_pos = unwrapped.last_robot_a_pos
        robot_b_pos = unwrapped.last_robot_b_pos
    box_xy = _first_xy(box_pos)
    robot_a_xy = _first_xy(robot_a_pos)
    robot_b_xy = _first_xy(robot_b_pos)
    goal_xy = _first_xy(unwrapped.goal_pos)
    box_yaw = float(getattr(unwrapped.cfg, "box_start_yaw", pi / 2.0))
    return SceneGraph(
        scene_id="manual_two_g1_box_push_live_isaac_sonic_adapter",
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
            "runtime": "live_isaac_sonic_adapter",
            "action_mode": str(getattr(unwrapped.cfg, "action_mode", "unknown")),
        },
    )


def _first_xy(tensor: Any) -> tuple[float, float]:
    value = tensor.detach().float()
    if value.ndim == 2:
        value = value[0]
    value = value.cpu()
    return (float(value[0].item()), float(value[1].item()))


def _mean_float(value: Any) -> float:
    return float(value.detach().float().mean().cpu().item())


def _mode_code(command: SonicCommand | None) -> int:
    return int(command.mode) if command is not None else -1


def _first_active_skill(active: dict[str, SonicCommand | None]) -> str:
    for command in active.values():
        if command is not None:
            return command.skill_node
    return "none"


def _first_active_phase(active: dict[str, SonicCommand | None]) -> str:
    for command in active.values():
        if command is not None:
            return command.phase_id or command.phase
    return "none"


def _skill_code(node_name: str) -> int:
    codes = {
        "none": 0,
        "approach_formation": 1,
        "establish_contact": 2,
        "pre_align_object": 3,
        "transport_object_path": 4,
        "release_and_exit": 5,
    }
    return codes.get(node_name, -1)


def _phase_code(phase_id: str) -> int:
    codes = {
        "none": 0,
        "approach_formation": 1,
        "establish_contact": 2,
        "lock_initial_hand_pose": 3,
        "pre_align_object": 4,
        "transport_object_path": 5,
        "release_and_exit": 6,
    }
    return codes.get(phase_id.split(".", 1)[0], -1)


def _torch():
    import torch

    return torch
