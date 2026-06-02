"""Differentiable M2 proxy for training the current Isaac box-push action.

The Isaac M2 task currently exposes a 4D proxy action: agent A proposes a
planar push, agent B proposes a planar push, and the box receives their averaged
velocity. This module trains that action surface before replacing it with true
G1 command tracking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from coevolve.algorithms.halypo import flatten_tensors, gradients_to_vector, halypo_projection


@dataclass
class M2ProxyRunResult:
    """Summary metrics for one M2 proxy training run."""

    method: str
    initial_team_return: float
    final_team_return: float
    initial_v: float
    final_v: float
    final_distance: float
    final_x: float
    final_y: float
    mean_update_alignment: float
    conflict_rate: float
    constraint_violation_rate: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


def action_from_theta(theta: torch.Tensor) -> torch.Tensor:
    """Map unconstrained parameters to the M2 4D bounded action."""

    return torch.tanh(theta)


def _rollout(
    theta: torch.Tensor,
    *,
    horizon: int,
    dt: float,
    goal: torch.Tensor,
    success_tolerance: float,
    velocity_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action = action_from_theta(theta).reshape(2, 2)
    box = torch.zeros(2, dtype=theta.dtype, device=theta.device)
    prev_x = box[0]
    trajectory = [box]
    rewards = []
    mismatch_total = torch.zeros((), dtype=theta.dtype, device=theta.device)

    for _ in range(horizon):
        push_a = action[0]
        push_b = action[1]
        mean_push = 0.5 * (push_a + push_b)
        mismatch = torch.linalg.norm(push_a - push_b)
        damping = torch.clamp(1.0 - 0.25 * mismatch, min=0.25)
        velocity = velocity_scale * mean_push * damping
        box = box + dt * velocity
        distance = torch.linalg.norm(goal - box)
        progress = box[0] - prev_x
        reward = 2.0 * progress - 0.1 * distance - 0.02 * mismatch
        reward = reward + torch.sigmoid(20.0 * (success_tolerance - distance))
        rewards.append(reward)
        mismatch_total = mismatch_total + mismatch.square()
        prev_x = box[0]
        trajectory.append(box)

    return torch.stack(trajectory), action, torch.stack(rewards), mismatch_total


def _objective(
    theta: torch.Tensor,
    *,
    horizon: int,
    dt: float,
    goal: torch.Tensor,
    success_tolerance: float,
    velocity_scale: float,
) -> torch.Tensor:
    trajectory, action, rewards, mismatch_total = _rollout(
        theta,
        horizon=horizon,
        dt=dt,
        goal=goal,
        success_tolerance=success_tolerance,
        velocity_scale=velocity_scale,
    )
    final_distance = torch.linalg.norm(goal - trajectory[-1])
    effort = action.square().sum()
    return rewards.sum() - 0.5 * final_distance - 0.01 * effort - 0.001 * mismatch_total


def _fields(
    theta: torch.Tensor,
    *,
    horizon: int,
    dt: float,
    goal: torch.Tensor,
    success_tolerance: float,
    velocity_scale: float,
    conflict_strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    team_objective = _objective(
        theta,
        horizon=horizon,
        dt=dt,
        goal=goal,
        success_tolerance=success_tolerance,
        velocity_scale=velocity_scale,
    )
    local_1 = team_objective + conflict_strength * theta[0] * theta[1]
    local_2 = team_objective - conflict_strength * theta[2] * theta[3]
    grad_1 = gradients_to_vector(local_1, [theta], create_graph=True)[:2]
    grad_2 = gradients_to_vector(local_2, [theta], create_graph=True)[2:]
    u_ind = torch.cat([grad_1, grad_2])
    u_team = gradients_to_vector(team_objective, [theta], create_graph=True)
    return u_ind, u_team, team_objective


def _measure(
    theta: torch.Tensor,
    *,
    horizon: int,
    dt: float,
    goal: torch.Tensor,
    success_tolerance: float,
    velocity_scale: float,
) -> tuple[float, float, float, float, float, list[float]]:
    trajectory, action, rewards, mismatch_total = _rollout(
        theta,
        horizon=horizon,
        dt=dt,
        goal=goal,
        success_tolerance=success_tolerance,
        velocity_scale=velocity_scale,
    )
    final_distance = float(torch.linalg.norm(goal - trajectory[-1]).detach())
    team_return = float(
        _objective(
            theta,
            horizon=horizon,
            dt=dt,
            goal=goal,
            success_tolerance=success_tolerance,
            velocity_scale=velocity_scale,
        ).detach()
    )
    mismatch = float(mismatch_total.detach())
    final_x = float(trajectory[-1, 0].detach())
    final_y = float(trajectory[-1, 1].detach())
    action_list = [float(value) for value in action.reshape(-1).detach()]
    _ = rewards
    return team_return, final_distance, mismatch, final_x, final_y, action_list


def _step(
    theta: torch.Tensor,
    *,
    method: str,
    lr: float,
    horizon: int,
    dt: float,
    goal: torch.Tensor,
    success_tolerance: float,
    velocity_scale: float,
    conflict_strength: float,
    sigma: float,
    eps: float,
) -> dict[str, float]:
    theta.requires_grad_(True)
    u_ind, u_team, team_objective = _fields(
        theta,
        horizon=horizon,
        dt=dt,
        goal=goal,
        success_tolerance=success_tolerance,
        velocity_scale=velocity_scale,
        conflict_strength=conflict_strength,
    )
    if method == "independent":
        gap = u_ind - u_team
        v = 0.5 * torch.dot(gap, gap)
        h = flatten_tensors(torch.autograd.grad(v, [theta], retain_graph=True))
        direction = u_ind.detach()
        lhs = torch.dot(h, direction)
        rhs = -sigma * v.detach()
    elif method == "halypo":
        projection = halypo_projection(u_ind, u_team, [theta], sigma=sigma, eps=eps)
        v = projection.v
        direction = projection.direction
        lhs = projection.lhs
        rhs = projection.rhs
    else:
        raise ValueError(f"unknown method: {method}")

    with torch.no_grad():
        theta += lr * direction

    team_return, final_distance, mismatch, final_x, final_y, action_list = _measure(
        theta,
        horizon=horizon,
        dt=dt,
        goal=goal,
        success_tolerance=success_tolerance,
        velocity_scale=velocity_scale,
    )
    update_alignment = _safe_cos(direction.detach(), u_team.detach())
    field_alignment = _safe_cos(u_ind.detach(), u_team.detach())
    violation = torch.clamp(lhs - rhs, min=0.0)
    return {
        "team_return": team_return,
        "team_objective_before_update": float(team_objective.detach()),
        "final_distance": final_distance,
        "mismatch": mismatch,
        "final_x": final_x,
        "final_y": final_y,
        "v": float(v.detach()),
        "field_alignment": float(field_alignment.detach()),
        "update_alignment": float(update_alignment.detach()),
        "constraint_violation": float(violation.detach()),
        "action_a_x": action_list[0],
        "action_a_y": action_list[1],
        "action_b_x": action_list[2],
        "action_b_y": action_list[3],
    }


def run_m2_proxy_training(
    *,
    method: str,
    updates: int = 80,
    lr: float = 0.02,
    horizon: int = 180,
    dt: float = 1.0 / 30.0,
    sigma: float = 0.2,
    eps: float = 1.0e-10,
    conflict_strength: float = 0.4,
    velocity_scale: float = 0.47,
    goal_x: float = 2.0,
    success_tolerance: float = 0.2,
    initial_theta: tuple[float, float, float, float] = (0.1, 0.8, 0.1, -0.8),
) -> tuple[M2ProxyRunResult, list[dict[str, float]], list[float], list[float]]:
    """Train a constant two-agent M2 proxy push action."""

    theta = torch.tensor(initial_theta, dtype=torch.float64)
    goal = torch.tensor([goal_x, 0.0], dtype=torch.float64)
    history: list[dict[str, float]] = []
    for update_idx in range(updates):
        row = _step(
            theta,
            method=method,
            lr=lr,
            horizon=horizon,
            dt=dt,
            goal=goal,
            success_tolerance=success_tolerance,
            velocity_scale=velocity_scale,
            conflict_strength=conflict_strength,
            sigma=sigma,
            eps=eps,
        )
        row["update"] = float(update_idx)
        history.append(row)

    conflicts = sum(1 for row in history if row["update_alignment"] <= 0.0)
    violations = sum(1 for row in history if row["constraint_violation"] > 1.0e-7)
    policy_theta = [float(value) for value in theta.detach()]
    policy_action = [float(value) for value in action_from_theta(theta).detach()]
    result = M2ProxyRunResult(
        method=method,
        initial_team_return=history[0]["team_return"],
        final_team_return=history[-1]["team_return"],
        initial_v=history[0]["v"],
        final_v=history[-1]["v"],
        final_distance=history[-1]["final_distance"],
        final_x=history[-1]["final_x"],
        final_y=history[-1]["final_y"],
        mean_update_alignment=sum(row["update_alignment"] for row in history) / len(history),
        conflict_rate=conflicts / len(history),
        constraint_violation_rate=violations / len(history),
    )
    return result, history, policy_theta, policy_action


def save_m2_proxy_plot(histories: dict[str, list[dict[str, float]]], output_path: Path) -> None:
    """Save return, gap, alignment, and proxy final-position curves."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    flat_axes = axes.reshape(-1)
    for method, history in histories.items():
        updates = [row["update"] for row in history]
        flat_axes[0].plot(updates, [row["team_return"] for row in history], label=method)
        flat_axes[1].plot(updates, [row["v"] for row in history], label=method)
        flat_axes[2].plot(updates, [row["update_alignment"] for row in history], label=method)
        flat_axes[3].plot([row["final_x"] for row in history], [row["final_y"] for row in history], label=method)

    flat_axes[0].set_title("M2 proxy team return")
    flat_axes[0].set_xlabel("update")
    flat_axes[1].set_title("Rationality gap V")
    flat_axes[1].set_xlabel("update")
    flat_axes[1].set_yscale("log")
    flat_axes[2].set_title("Update-team alignment")
    flat_axes[2].set_xlabel("update")
    flat_axes[2].set_ylim(-1.05, 1.05)
    flat_axes[3].set_title("Final box position over training")
    flat_axes[3].set_xlabel("x")
    flat_axes[3].set_ylabel("y")
    flat_axes[3].scatter([2.0], [0.0], marker="*", s=120, label="goal")
    flat_axes[3].axis("equal")
    for axis in flat_axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
