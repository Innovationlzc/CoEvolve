"""PCGrad baseline utilities.

PCGrad projects one task gradient away from another when their dot product is
negative, then averages the adjusted gradients. This is a small deterministic
kernel for comparison against HALyPO-style projection.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


Tensor = torch.Tensor


def pcgrad_project(gradients: Sequence[Tensor], *, eps: float = 1.0e-12) -> list[Tensor]:
    """Return conflict-projected gradients without mutating the inputs."""

    if not gradients:
        raise ValueError("gradients must not be empty.")
    shape = gradients[0].shape
    if any(gradient.shape != shape for gradient in gradients):
        raise ValueError("all gradients must have the same shape.")

    projected = [gradient.clone() for gradient in gradients]
    references = [gradient.detach() for gradient in gradients]
    for i, grad_i in enumerate(projected):
        for j, grad_j in enumerate(references):
            if i == j:
                continue
            dot = torch.dot(grad_i.reshape(-1), grad_j.reshape(-1))
            if dot < 0.0:
                denom = torch.dot(grad_j.reshape(-1), grad_j.reshape(-1)) + eps
                projected[i] = grad_i - dot / denom * grad_j
                grad_i = projected[i]
    return projected


def pcgrad_direction(gradients: Sequence[Tensor], *, eps: float = 1.0e-12) -> Tensor:
    """Return the averaged PCGrad direction."""

    projected = pcgrad_project(gradients, eps=eps)
    return torch.stack(projected, dim=0).mean(dim=0)
