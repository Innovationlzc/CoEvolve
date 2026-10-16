"""Geometry-aware payload attitude metrics without MuJoCo runtime dependencies."""

from __future__ import annotations

import numpy as np


def payload_support_face_attitude(
    rotation: np.ndarray,
    half_extents: np.ndarray,
) -> tuple[float, float, str]:
    """Measure attitude without distinguishing geometrically identical box faces."""

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    half_extents = np.asarray(half_extents, dtype=np.float64).reshape(3)
    long_axis = int(np.argmax(half_extents))
    cross_axes = [axis for axis in range(3) if axis != long_axis]
    equivalent_cross_faces = bool(
        np.isclose(
            half_extents[cross_axes[0]],
            half_extents[cross_axes[1]],
            rtol=1.0e-3,
            atol=1.0e-6,
        )
    )
    candidate_axes = cross_axes if equivalent_cross_faces else [2]
    support_axis = max(candidate_axes, key=lambda axis: abs(rotation[2, axis]))
    upward_alignment = float(np.clip(abs(rotation[2, support_axis]), 0.0, 1.0))
    tilt_deg = float(np.degrees(np.arccos(upward_alignment)))
    tangent_axes = [axis for axis in range(3) if axis != support_axis]
    z_range_m = 2.0 * float(
        sum(abs(rotation[2, axis]) * half_extents[axis] for axis in tangent_axes)
    )
    return tilt_deg, z_range_m, ("x", "y", "z")[support_axis]
