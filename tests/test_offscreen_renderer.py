from types import SimpleNamespace

import numpy as np
import pytest

from praxis_coop.sonic_mujoco.offscreen_renderer import _apply_snapshot


def _model_and_data():
    model = SimpleNamespace(
        geom_rgba=np.zeros((3, 4), dtype=np.float32),
        site_rgba=np.zeros((2, 4), dtype=np.float32),
    )
    data = SimpleNamespace(
        qpos=np.zeros(5, dtype=np.float64),
        qvel=np.ones(4, dtype=np.float64),
    )
    return model, data


def test_offscreen_renderer_applies_atomic_visual_snapshot():
    model, data = _model_and_data()
    snapshot = {
        "qpos": np.arange(5, dtype=np.float64),
        "geom_rgba": np.full((3, 4), 0.5, dtype=np.float32),
        "site_rgba": np.full((2, 4), 0.25, dtype=np.float32),
    }

    _apply_snapshot(model, data, snapshot)

    assert np.array_equal(data.qpos, snapshot["qpos"])
    assert np.array_equal(data.qvel, np.zeros(4))
    assert np.array_equal(model.geom_rgba, snapshot["geom_rgba"])
    assert np.array_equal(model.site_rgba, snapshot["site_rgba"])


def test_offscreen_renderer_rejects_nonfinite_or_mismatched_state():
    model, data = _model_and_data()
    snapshot = {
        "qpos": np.asarray([0.0, 1.0, np.nan, 3.0, 4.0]),
        "geom_rgba": np.zeros((3, 4), dtype=np.float32),
        "site_rgba": np.zeros((2, 4), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="non-finite"):
        _apply_snapshot(model, data, snapshot)

    snapshot["qpos"] = np.zeros(4)
    with pytest.raises(ValueError, match="qpos shape"):
        _apply_snapshot(model, data, snapshot)
