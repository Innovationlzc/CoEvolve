"""M0 two-agent differentiable game for HALyPO validation.

The team objective wants ``theta = [x, y]`` to move toward the origin. The
independent field is a rotational field, so plain independent ascent cycles
instead of reducing the rationality gap. HALyPO adds the smallest correction
needed to satisfy the Lyapunov decrease constraint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from coevolve.algorithms.halypo import gradients_to_vector, halypo_projection


@dataclass
class ToyRunResult:
    """Summary metrics for one toy run."""

    method: str
    initial_v: float
    final_v: float
    mean_update_alignment: float
    conflict_rate: float
    constraint_violation_rate: float
    final_theta_norm: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _fields(theta: torch.Tensor, rotation: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent and team rationality fields for the toy game."""

    x, y = theta[0], theta[1]
    j1 = rotation * x * y
    j2 = -rotation * x * y
    j_team = -0.5 * (x.square() + y.square())
    u_ind = torch.stack(
        [
            gradients_to_vector(j1, [theta], create_graph=True)[0],
            gradients_to_vector(j2, [theta], create_graph=True)[1],
        ]
    )
    u_team = gradients_to_vector(j_team, [theta], create_graph=True)
    return u_ind, u_team


def _safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.dot(a, b) / (a.norm() * b.norm() + eps)


def _step(
    theta: torch.Tensor,
    *,
    method: str,
    lr: float,
    rotation: float,
    sigma: float,
    eps: float,
) -> dict[str, float]:
    theta.requires_grad_(True)
    u_ind, u_team = _fields(theta, rotation)
    if method == "independent":
        gap = u_ind - u_team
        v = 0.5 * torch.dot(gap, gap)
        h = torch.autograd.grad(v, [theta], retain_graph=True)[0]
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

    update_alignment = _safe_cos(direction.detach(), u_team.detach())
    field_alignment = _safe_cos(u_ind.detach(), u_team.detach())
    violation = torch.clamp(lhs - rhs, min=0.0)
    return {
        "x": float(theta.detach()[0]),
        "y": float(theta.detach()[1]),
        "v": float(v.detach()),
        "field_alignment": float(field_alignment.detach()),
        "update_alignment": float(update_alignment.detach()),
        "constraint_violation": float(violation.detach()),
        "constraint_lhs": float(lhs.detach()),
        "constraint_rhs": float(rhs.detach()),
    }


def run_toy_game(
    *,
    method: str,
    steps: int = 300,
    lr: float = 0.04,
    rotation: float = 1.25,
    sigma: float = 0.5,
    eps: float = 1.0e-8,
    initial_theta: tuple[float, float] = (2.0, 0.75),
) -> tuple[ToyRunResult, list[dict[str, float]]]:
    """Run the M0 game and return summary plus per-step history."""

    theta = torch.tensor(initial_theta, dtype=torch.float64)
    history: list[dict[str, float]] = []
    for step_idx in range(steps):
        row = _step(theta, method=method, lr=lr, rotation=rotation, sigma=sigma, eps=eps)
        row["step"] = float(step_idx)
        history.append(row)

    initial_v = history[0]["v"]
    final_v = history[-1]["v"]
    mean_update_alignment = sum(row["update_alignment"] for row in history) / len(history)
    conflicts = sum(1 for row in history if row["update_alignment"] <= 0.0)
    violations = sum(1 for row in history if row["constraint_violation"] > 1.0e-7)
    result = ToyRunResult(
        method=method,
        initial_v=initial_v,
        final_v=final_v,
        mean_update_alignment=mean_update_alignment,
        conflict_rate=conflicts / len(history),
        constraint_violation_rate=violations / len(history),
        final_theta_norm=float(theta.norm().detach()),
    )
    return result, history


def save_plot(histories: dict[str, list[dict[str, float]]], output_path: Path) -> None:
    """Save V, alignment, and trajectory curves for M0."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for method, history in histories.items():
        steps = [row["step"] for row in history]
        axes[0].plot(steps, [row["v"] for row in history], label=method)
        axes[1].plot(steps, [row["update_alignment"] for row in history], label=method)
        axes[2].plot([row["x"] for row in history], [row["y"] for row in history], label=method)

    axes[0].set_title("Rationality gap V")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("V")
    axes[0].set_yscale("log")
    axes[1].set_title("Update-team alignment")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("cos(direction, u_team)")
    axes[1].set_ylim(-1.05, 1.05)
    axes[2].set_title("Parameter trajectory")
    axes[2].set_xlabel("theta[0]")
    axes[2].set_ylabel("theta[1]")
    axes[2].axis("equal")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

