import math

import numpy as np
import pytest

from praxis_coop.sonic_mujoco.payload_attitude import payload_support_face_attitude


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)),
        dtype=np.float64,
    )


def test_square_cross_section_accepts_equivalent_quarter_turn_support_face():
    tilt, z_range, axis = payload_support_face_attitude(
        _rotation_x(90.0),
        np.asarray((0.9, 0.05, 0.05)),
    )

    assert tilt == pytest.approx(0.0, abs=1.0e-8)
    assert z_range == pytest.approx(0.0, abs=1.0e-8)
    assert axis == "y"


def test_square_cross_section_still_rejects_edge_balancing_attitude():
    tilt, z_range, axis = payload_support_face_attitude(
        _rotation_x(45.0),
        np.asarray((0.9, 0.05, 0.05)),
    )

    assert tilt == pytest.approx(45.0)
    assert z_range == pytest.approx(math.sqrt(0.005))
    assert axis in {"y", "z"}


def test_rectangular_cross_section_retains_designated_top_face():
    tilt, z_range, axis = payload_support_face_attitude(
        _rotation_x(90.0),
        np.asarray((0.9, 0.10, 0.05)),
    )

    assert tilt == pytest.approx(90.0)
    assert z_range == pytest.approx(0.20)
    assert axis == "z"
