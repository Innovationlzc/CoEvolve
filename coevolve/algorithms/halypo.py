"""HALyPO projection utilities.

The kernel operates on flattened gradient fields. In a training loop, compute
``u_ind`` and ``u_team`` with ``create_graph=True`` before calling
``halypo_projection`` so the Lyapunov normal ``h = grad V`` is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class HALyPOProjection:
    """Result of one HALyPO direction projection."""

    direction: Tensor
    h: Tensor
    v: Tensor
    psi: Tensor
    lambda_star: Tensor
    lhs: Tensor
    rhs: Tensor

    @property
    def violation(self) -> Tensor:
        """Positive value means the Lyapunov decrease constraint is violated."""

        return torch.clamp(self.lhs - self.rhs, min=0.0)


def flatten_tensors(tensors: Iterable[Tensor]) -> Tensor:
    """Flatten and concatenate a sequence of tensors."""

    flat = [tensor.reshape(-1) for tensor in tensors if tensor is not None]
    if not flat:
        raise ValueError("Cannot flatten an empty tensor sequence.")
    return torch.cat(flat)


def gradients_to_vector(
    loss_or_objective: Tensor,
    parameters: Sequence[Tensor],
    *,
    create_graph: bool,
    retain_graph: bool = True,
) -> Tensor:
    """Differentiate a scalar with respect to parameters and flatten gradients."""

    grads = torch.autograd.grad(
        loss_or_objective,
        parameters,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=False,
    )
    return flatten_tensors(grads)


def project_direction(
    h: Tensor,
    u_ind: Tensor,
    v: Tensor,
    *,
    sigma: float = 0.5,
    eps: float = 1.0e-8,
) -> HALyPOProjection:
    """Project ``u_ind`` into the Lyapunov-stable half-space.

    The constraint is ``dot(h, d) <= -sigma * V``. The returned direction is the
    minimum-change projection of ``u_ind`` onto that half-space.
    """

    if h.shape != u_ind.shape:
        raise ValueError(f"shape mismatch: h={tuple(h.shape)}, u_ind={tuple(u_ind.shape)}")
    if v.numel() != 1:
        raise ValueError("v must be a scalar tensor.")

    v_detached = v.detach()
    psi = torch.dot(h, u_ind.detach()) + sigma * v_detached
    denom = torch.dot(h, h) + eps
    lambda_star = torch.clamp(psi / denom, min=0.0)
    direction = u_ind.detach() - lambda_star * h
    lhs = torch.dot(h, direction)
    rhs = -sigma * v_detached
    return HALyPOProjection(
        direction=direction,
        h=h,
        v=v,
        psi=psi,
        lambda_star=lambda_star,
        lhs=lhs,
        rhs=rhs,
    )


def halypo_projection(
    u_ind: Tensor,
    u_team: Tensor,
    parameters: Sequence[Tensor],
    *,
    sigma: float = 0.5,
    eps: float = 1.0e-8,
    create_graph: bool = False,
) -> HALyPOProjection:
    """Compute ``V``, ``h = grad V``, and the projected HALyPO direction."""

    if u_ind.shape != u_team.shape:
        raise ValueError(f"shape mismatch: u_ind={tuple(u_ind.shape)}, u_team={tuple(u_team.shape)}")
    if not parameters:
        raise ValueError("parameters must not be empty.")

    gap = u_ind - u_team
    v = 0.5 * torch.dot(gap, gap)
    h_parts = torch.autograd.grad(v, parameters, create_graph=create_graph, retain_graph=True, allow_unused=False)
    h = flatten_tensors(h_parts)
    return project_direction(h, u_ind, v, sigma=sigma, eps=eps)

