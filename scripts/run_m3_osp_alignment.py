#!/usr/bin/env python3
"""Run the CoEvolve M3 OSP-Alignment-Lite Isaac task."""

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
    parser.add_argument("--task", default="CoEvolve-OSP-Alignment-Lite-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results/m3_osp_alignment_lite"))
    parser.add_argument("--stop-on-success", action="store_true")
    add_launcher_args(parser)
    return setup_preset_cli(parser)


def _mean_float(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu().item())


def _write_outputs(args_cli: argparse.Namespace, history: list[dict[str, float]]) -> None:
    if not history:
        return
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args_cli.output_dir / "history.csv"
    metrics_path = args_cli.output_dir / "metrics.json"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    metrics = {
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "steps": len(history),
        "final": history[-1],
        "note": (
            "M3 OSP-Alignment-Lite smoke. This has a paper-like OSP scene, 11D command surface, "
            "G1 wrist target markers, contact scores, and a temporary root-stabilized arm tracker. "
            "It is not yet true G1 walking/WBC or physical hand-object manipulation."
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[M3] wrote {metrics_path}", flush=True)
    print(f"[M3] wrote {history_path}", flush=True)


def _scripted_action(env: gym.Env) -> torch.Tensor:
    unwrapped = env.unwrapped
    action = torch.zeros(env.action_space.shape, device=unwrapped.device)
    action[:, 0] = 0.92
    action[:, 1] = -0.15 * torch.tanh(unwrapped.object.data.root_pos_w.torch[:, 1] - unwrapped.scene.env_origins[:, 1])
    yaw_error = -unwrapped.object_yaw
    action[:, 2] = torch.clamp(1.2 * yaw_error, min=-1.0, max=1.0)
    action[:, 3] = 0.0
    action[:, 4] = 0.15
    action[:, 5] = 0.0
    action[:, 6] = 0.0
    action[:, 7] = 0.0
    action[:, 8] = 0.0
    action[:, 9] = 0.0
    action[:, 10] = 0.0
    return action


def main() -> None:
    args_cli, hydra_args = parse_args()
    sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

    torch.manual_seed(7)
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    history = []
    first_success_step = None
    best_distance = float("inf")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        env.reset()
        sim = env.unwrapped.sim
        for step in range(args_cli.max_steps):
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break
            with torch.inference_mode():
                _, reward, terminated, truncated, _ = env.step(_scripted_action(env))
                object_pos = env.unwrapped.object.data.root_pos_w.torch - env.unwrapped.scene.env_origins
                distance = env.unwrapped.last_distance
                yaw_error = env.unwrapped.last_yaw_error
                success_event_rate = _mean_float(env.unwrapped.last_success)
                metric_object_pos = env.unwrapped.last_object_pos if success_event_rate > 0.0 else object_pos
                metric_contact_scores = (
                    env.unwrapped.last_contact_scores if success_event_rate > 0.0 else env.unwrapped.contact_scores
                )
                contact = metric_contact_scores[:, :2].mean(dim=-1)
                best_distance = min(best_distance, float(distance.detach().min().cpu().item()))
                if first_success_step is None and success_event_rate > 0.0:
                    first_success_step = step + 1
                row = {
                    "step": step + 1,
                    "mean_reward": _mean_float(reward),
                    "mean_object_x": _mean_float(metric_object_pos[:, 0]),
                    "mean_object_y": _mean_float(metric_object_pos[:, 1]),
                    "mean_distance": _mean_float(distance),
                    "mean_yaw_error": _mean_float(yaw_error),
                    "mean_g1_contact": _mean_float(contact),
                    "success_event_rate": success_event_rate,
                    "done_rate": _mean_float(terminated | truncated),
                    "first_success_step": float(first_success_step or 0),
                    "best_distance": best_distance,
                }
                history.append(row)
                if args_cli.log_every > 0 and row["step"] % args_cli.log_every == 0:
                    print(
                        "[M3] "
                        f"step={row['step']} "
                        f"x={row['mean_object_x']:.3f} "
                        f"dist={row['mean_distance']:.3f} "
                        f"yaw={row['mean_yaw_error']:.3f} "
                        f"contact={row['mean_g1_contact']:.3f} "
                        f"success_event={row['success_event_rate']:.2f}",
                        flush=True,
                    )
            if args_cli.stop_on_success and first_success_step is not None:
                break
        _write_outputs(args_cli, history)
        env.close()


if __name__ == "__main__":
    main()
