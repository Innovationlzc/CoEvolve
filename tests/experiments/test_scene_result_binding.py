from __future__ import annotations

import pytest

from coevolve_experiments.reporting.scene_result_binding import (
    SceneResultBindingError,
    resolve_scene_result_binding,
)


def _report() -> dict[str, object]:
    return {
        "scene_id": "base", "mode": "execution", "errors": [], "warnings": [],
        "execution_evidence_fingerprint": "a" * 64,
        "evidence_receipts": [{
            "check": "simulator_integrity",
            "raw_artifacts": [
                {"artifact_type": "compiled_model", "relative_path": "model.mjb", "sha256": "b" * 64, "size_bytes": 10},
                {"artifact_type": "config", "relative_path": "config.json", "sha256": "c" * 64, "size_bytes": 10},
            ],
        }],
    }


def test_scene_result_binding_uses_preflight_compiled_model() -> None:
    result = resolve_scene_result_binding(
        [_report()], scene_id="base", scene_fingerprint="d" * 64,
        scene_realization_sha256="e" * 64,
    )
    assert result.compiled_model_sha256 == "b" * 64
    assert result.scene_execution_evidence_fingerprint == "a" * 64


def test_scene_result_binding_fails_closed_on_ambiguous_compiled_model() -> None:
    report = _report()
    report["evidence_receipts"][0]["raw_artifacts"].append(  # type: ignore[index]
        {"artifact_type": "compiled_model", "relative_path": "other.mjb", "sha256": "f" * 64, "size_bytes": 10}
    )
    with pytest.raises(SceneResultBindingError, match="exactly once"):
        resolve_scene_result_binding(
            [report], scene_id="base", scene_fingerprint="d" * 64,
            scene_realization_sha256="e" * 64,
        )
