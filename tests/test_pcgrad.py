import pytest
import torch

from coevolve.algorithms.pcgrad import pcgrad_direction, pcgrad_project


def test_pcgrad_removes_negative_pairwise_component() -> None:
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 1.0])

    p1, p2 = pcgrad_project([g1, g2], eps=0.0)

    assert torch.dot(p1, g2) >= -1.0e-7
    assert torch.dot(p2, g1) >= -1.0e-7


def test_pcgrad_keeps_non_conflicting_gradients() -> None:
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([0.0, 1.0])

    p1, p2 = pcgrad_project([g1, g2])

    torch.testing.assert_close(p1, g1)
    torch.testing.assert_close(p2, g2)
    torch.testing.assert_close(pcgrad_direction([g1, g2]), torch.tensor([0.5, 0.5]))


def test_pcgrad_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        pcgrad_project([torch.ones(2), torch.ones(3)])
