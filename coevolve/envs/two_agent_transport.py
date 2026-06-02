"""M1 differentiable two-agent transport validation.

This is a small policy-gradient proxy for cooperative transport. Two agents
choose constant 2D velocity commands. The object follows their average command,
while disagreement between commands acts like object tilt/drag. Each agent's
local objective includes a small heterogeneous rotational term, creating a
rationality gap against the true team objective. HALyPO should reduce that gap
while preserving transport performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from coevolve.algorithms.halypo import flatten_tensors, gradients_to_vector, halypo_projection


@dataclass
class TransportRunResult:
    """Summary metrics for one M1 training run."""

    method: str
    initial_team_return: float
    final_team_return: float
    initial_v: float
    final_v: float
    final_distance: float
    mean_update_alignment: float
    conflict_rate: float
    constraint_violation_rate: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


def _rollout(
    theta: torch.Tensor,
    *,
    horizon: int,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll out object motion and return trajectory, actions, disagreement."""

    actions = torch.tanh(theta).reshape(2, 2)
    obj = torch.zeros(2, dtype=theta.dtype, device=theta.device)
    trajectory = [obj]
    disagreement_cost = torch.zeros((), dtype=theta.dtype, device=theta.device)
    for _ in range(horizon):
        mean_action = actions.mean(dim=0)
        mismatch = actions[0] - actions[1]
        drag = 0.08 * mismatch.square().sum()
        obj = obj + dt * (mean_action * (1.0 - drag).clamp(min=0.2))
        disagreement_cost = disagreement_cost + mismatch.square().sum()
        trajectory.append(obj)
    return torch.stack(trajectory), actions, disagreement_cost


def _objective(theta: torch.Tensor, *, target: torch.Tensor, horizon: int, dt: float) -> torch.Tensor:
    trajectory, actions, disagreement_cost = _rollout(theta, horizon=horizon, dt=dt)
    final_distance = (trajectory[-1] - target).norm()
    progress = trajectory[-1][0] - trajectory[0][0]
    effort = actions.square().sum()
    return -final_distance + 0.05 * progress - 0.01 * effort - 0.002 * disagreement_cost


def _fields(
    theta: torch.Tensor,
    *,
    horizon: int,
    dt: float,
    goal: torch.Tensor,
    conflict_strength: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute independent and team rationality fields."""

    team_objective = _objective(theta, target=goal, horizon=horizon, dt=dt)
    local_1 = team_objective + conflict_strength * theta[0] * theta[1]
    local_2 = team_objective - conflict_strength * theta[2] * theta[3]

    grad_1 = gradients_to_vector(local_1, [theta], create_graph=True)[:2]
    grad_2 = gradients_to_vector(local_2, [theta], create_graph=True)[2:]
    u_ind = torch.cat([grad_1, grad_2])
    u_team = gradients_to_vector(team_objective, [theta], create_graph=True)
    return u_ind, u_team, team_objective


def _measure(theta: torch.Tensor, *, horizon: int, dt: float, goal: torch.Tensor) -> tuple[float, float, float, float, float]:
    trajectory, actions, disagreement_cost = _rollout(theta, horizon=horizon, dt=dt)
    final_distance = float((trajectory[-1] - goal).norm().detach())
    team_return = float(_objective(theta, target=goal, horizon=horizon, dt=dt).detach())
    mismatch = float(disagreement_cost.detach())
    final_x = float(trajectory[-1, 0].detach())
    final_y = float(trajectory[-1, 1].detach())
    _ = actions
    return team_return, final_distance, mismatch, final_x, final_y


def _step(
    theta: torch.Tensor,
    *,
    method: str,
    lr: float,
    horizon: int,
    dt: float,
    goal: torch.Tensor,
    conflict_strength: float,
    sigma: float,
    eps: float,
) -> dict[str, float]:
    theta.requires_grad_(True)
    u_ind, u_team, team_objective = _fields(
        theta, horizon=horizon, dt=dt, goal=goal, conflict_strength=conflict_strength
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

    team_return, final_distance, mismatch, final_x, final_y = _measure(theta, horizon=horizon, dt=dt, goal=goal)
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
    }


def run_transport_training(
    *,
    method: str,
    updates: int = 60,
    lr: float = 0.02,
    horizon: int = 35,
    dt: float = 0.08,
    sigma: float = 0.5,
    eps: float = 1.0e-8,
    conflict_strength: float = 0.8,
    initial_theta: tuple[float, float, float, float] = (0.05, 1.0, 0.05, -1.0),
) -> tuple[TransportRunResult, list[dict[str, float]]]:
    """Train the M1 transport proxy and return summary plus per-update history."""

    theta = torch.tensor(initial_theta, dtype=torch.float64)
    goal = torch.tensor([2.0, 0.0], dtype=torch.float64)
    history: list[dict[str, float]] = []
    for update_idx in range(updates):
        row = _step(
            theta,
            method=method,
            lr=lr,
            horizon=horizon,
            dt=dt,
            goal=goal,
            conflict_strength=conflict_strength,
            sigma=sigma,
            eps=eps,
        )
        row["update"] = float(update_idx)
        history.append(row)

    conflicts = sum(1 for row in history if row["update_alignment"] <= 0.0)
    violations = sum(1 for row in history if row["constraint_violation"] > 1.0e-7)
    result = TransportRunResult(
        method=method,
        initial_team_return=history[0]["team_return"],
        final_team_return=history[-1]["team_return"],
        initial_v=history[0]["v"],
        final_v=history[-1]["v"],
        final_distance=history[-1]["final_distance"],
        mean_update_alignment=sum(row["update_alignment"] for row in history) / len(history),
        conflict_rate=conflicts / len(history),
        constraint_violation_rate=violations / len(history),
    )
    return result, history


def save_transport_plot(histories: dict[str, list[dict[str, float]]], output_path: Path) -> None:
    """Save M1 return, gap, alignment, and final-object trajectory curves."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    flat_axes = axes.reshape(-1)
    for method, history in histories.items():
        updates = [row["update"] for row in history]
        flat_axes[0].plot(updates, [row["team_return"] for row in history], label=method)
        flat_axes[1].plot(updates, [row["v"] for row in history], label=method)
        flat_axes[2].plot(updates, [row["update_alignment"] for row in history], label=method)
        flat_axes[3].plot([row["final_x"] for row in history], [row["final_y"] for row in history], label=method)

    flat_axes[0].set_title("Team return")
    flat_axes[0].set_xlabel("update")
    flat_axes[1].set_title("Rationality gap V")
    flat_axes[1].set_xlabel("update")
    flat_axes[1].set_yscale("log")
    flat_axes[2].set_title("Update-team alignment")
    flat_axes[2].set_xlabel("update")
    flat_axes[2].set_ylim(-1.05, 1.05)
    flat_axes[3].set_title("Final object position over training")
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
