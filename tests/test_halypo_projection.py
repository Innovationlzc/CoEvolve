from __future__ import annotations

import torch

from coevolve.algorithms.halypo import gradients_to_vector, halypo_projection, project_direction


def test_project_direction_inactive_case_keeps_independent_direction() -> None:
    h = torch.tensor([1.0, 0.0])
    u_ind = torch.tensor([-2.0, 2.0])
    v = torch.tensor(1.0)

    result = project_direction(h, u_ind, v, sigma=0.5, eps=0.0)

    assert result.lambda_star.item() == 0.0
    torch.testing.assert_close(result.direction, u_ind)
    assert result.lhs <= result.rhs


def test_project_direction_active_case_satisfies_boundary() -> None:
    h = torch.tensor([1.0, 0.0])
    u_ind = torch.tensor([1.0, 2.0])
    v = torch.tensor(1.0)

    result = project_direction(h, u_ind, v, sigma=0.5, eps=0.0)

    assert result.lambda_star.item() > 0.0
    torch.testing.assert_close(result.direction, torch.tensor([-0.5, 2.0]))
    torch.testing.assert_close(result.lhs, result.rhs)


def test_halypo_projection_shape_and_constraint() -> None:
    theta = torch.tensor([2.0, 0.75], dtype=torch.float64, requires_grad=True)
    x, y = theta[0], theta[1]
    j1 = x * y
    j2 = -x * y
    j_team = -0.5 * (x.square() + y.square())
    u_ind = torch.stack(
        [
            gradients_to_vector(j1, [theta], create_graph=True)[0],
            gradients_to_vector(j2, [theta], create_graph=True)[1],
        ]
    )
    u_team = gradients_to_vector(j_team, [theta], create_graph=True)

    result = halypo_projection(u_ind, u_team, [theta], sigma=0.5, eps=1.0e-10)

    assert result.direction.shape == theta.shape
    assert result.h.shape == theta.shape
    assert result.v.item() > 0.0
    assert result.violation.item() < 1.0e-7


def test_halypo_projection_supports_create_graph() -> None:
    theta = torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True)
    x, y = theta[0], theta[1]
    u_ind = torch.stack(
        [
            gradients_to_vector(x * y, [theta], create_graph=True)[0],
            gradients_to_vector(-x * y, [theta], create_graph=True)[1],
        ]
    )
    u_team = gradients_to_vector(-0.5 * (x.square() + y.square()), [theta], create_graph=True)

    result = halypo_projection(u_ind, u_team, [theta], sigma=0.5, create_graph=True)
    probe = result.h.square().sum()
    grad = torch.autograd.grad(probe, [theta], allow_unused=False)[0]

    assert grad.shape == theta.shape
    assert torch.isfinite(grad).all()

