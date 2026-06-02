#!/usr/bin/env python3
"""Run the CoEvolve Two-G1 Box Push Lite Isaac task with random actions."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import coevolve.isaac_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, fold_preset_tokens, launch_simulation, resolve_task_config, setup_preset_cli

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="CoEvolve-TwoG1BoxPush-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("results/m2_two_g1_box_push"))
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--action-mode", choices=("proxy_box_velocity", "base_command"), default="proxy_box_velocity")
    parser.add_argument("--stop-on-success", action="store_true")
    add_launcher_args(parser)
    return setup_preset_cli(parser)


def _mean_float(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def _write_outputs(args_cli: argparse.Namespace, history: list[dict[str, float]]) -> None:
    if not history:
        return

    history_path = args_cli.output_dir / "history.csv"
    metrics_path = args_cli.output_dir / "metrics.json"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "task": args_cli.task,
        "action_mode": args_cli.action_mode,
        "num_envs": args_cli.num_envs,
        "steps": len(history),
        "final": history[-1],
        "note": (
            "M2 eval. proxy_box_velocity writes box velocity directly; "
            "base_command moves G1 base visuals first and pushes through a virtual contact adapter."
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[M2] wrote {metrics_path}", flush=True)
    print(f"[M2] wrote {history_path}", flush=True)


def main() -> None:
    args_cli, hydra_args = parse_args()
    sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

    torch.manual_seed(42)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    policy_action = None
    if args_cli.policy_path is not None:
        policy_data = json.loads(args_cli.policy_path.read_text(encoding="utf-8"))
        policy_action = torch.tensor(policy_data["action"], dtype=torch.float32)
        print(f"[M2] loaded policy action from {args_cli.policy_path}: {policy_data['action']}")

    env_cfg, _ = resolve_task_config(args_cli.task, "")
    history = []
    first_success_step = None
    best_distance = float("inf")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.action_mode = args_cli.action_mode
        env = gym.make(args_cli.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        env.reset()
        sim = env.unwrapped.sim
        steps = 0
        while steps < args_cli.max_steps:
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break
            with torch.inference_mode():
                if policy_action is None:
                    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    actions[:, 0] = 0.8
                    actions[:, 2] = 0.8
                    actions += 0.05 * (2 * torch.rand_like(actions) - 1)
                else:
                    actions = policy_action.to(env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                _, reward, terminated, truncated, _ = env.step(actions)
                box_pos = env.unwrapped.box.data.root_pos_w.torch - env.unwrapped.scene.env_origins
                goal_xy = env.unwrapped.goal_pos[:, 0:2]
                distance = torch.linalg.norm(goal_xy - box_pos[:, 0:2], dim=-1)
                success_event_rate = _mean_float(env.unwrapped.last_success)
                metric_box_pos = env.unwrapped.last_box_pos if success_event_rate > 0.0 else box_pos
                metric_distance = env.unwrapped.last_distance if success_event_rate > 0.0 else distance
                robot_a_pos = env.unwrapped.robot_a.data.root_pos_w.torch[:, :3] - env.unwrapped.scene.env_origins
                robot_b_pos = env.unwrapped.robot_b.data.root_pos_w.torch[:, :3] - env.unwrapped.scene.env_origins
                metric_robot_a_pos = env.unwrapped.last_robot_a_pos if success_event_rate > 0.0 else robot_a_pos
                metric_robot_b_pos = env.unwrapped.last_robot_b_pos if success_event_rate > 0.0 else robot_b_pos
                best_distance = min(best_distance, float(metric_distance.detach().min().cpu().item()))
                if first_success_step is None and success_event_rate > 0.0:
                    first_success_step = steps + 1
                row = {
                    "step": steps + 1,
                    "mean_reward": _mean_float(reward),
                    "mean_box_x": _mean_float(metric_box_pos[:, 0]),
                    "mean_box_y": _mean_float(metric_box_pos[:, 1]),
                    "mean_robot_a_x": _mean_float(metric_robot_a_pos[:, 0]),
                    "mean_robot_a_y": _mean_float(metric_robot_a_pos[:, 1]),
                    "mean_robot_a_z": _mean_float(metric_robot_a_pos[:, 2]),
                    "mean_robot_b_x": _mean_float(metric_robot_b_pos[:, 0]),
                    "mean_robot_b_y": _mean_float(metric_robot_b_pos[:, 1]),
                    "mean_robot_b_z": _mean_float(metric_robot_b_pos[:, 2]),
                    "mean_distance": _mean_float(metric_distance),
                    "success_rate": _mean_float(metric_distance < env.unwrapped.cfg.success_tolerance),
                    "success_event_rate": success_event_rate,
                    "done_rate": _mean_float(terminated | truncated),
                    "first_success_step": float(first_success_step or 0),
                    "best_distance": best_distance,
                }
                history.append(row)
                if args_cli.log_every > 0 and row["step"] % args_cli.log_every == 0:
                    print(
                        "[M2] "
                        f"step={row['step']} "
                        f"x={row['mean_box_x']:.3f} "
                        f"dist={row['mean_distance']:.3f} "
                        f"success_event={row['success_event_rate']:.2f}"
                    )
            steps += 1
            if args_cli.stop_on_success and first_success_step is not None:
                break
        _write_outputs(args_cli, history)
        env.close()


if __name__ == "__main__":
    main()
