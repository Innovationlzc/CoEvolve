"""Resolve terminal-evidence identities from a validated suite scene report."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SceneResultBindingError(ValueError):
    """A suite execution report cannot bind one run's simulator evidence."""


@dataclass(frozen=True)
class SceneResultBinding:
    scene_fingerprint: str
    scene_realization_sha256: str
    scene_execution_evidence_fingerprint: str
    compiled_model_sha256: str

    def as_expectation(self) -> dict[str, str]:
        return {
            "scene_fingerprint": self.scene_fingerprint,
            "scene_realization_sha256": self.scene_realization_sha256,
            "scene_execution_evidence_fingerprint": self.scene_execution_evidence_fingerprint,
            "compiled_model_sha256": self.compiled_model_sha256,
        }


def resolve_scene_result_binding(
    reports: Sequence[Mapping[str, Any]], *, scene_id: str,
    scene_fingerprint: str, scene_realization_sha256: str,
) -> SceneResultBinding:
    """Extract the compiled-model pin from exactly one execution-ready report."""

    if not _sha(scene_fingerprint) or not _sha(scene_realization_sha256):
        raise SceneResultBindingError("RunSpec/scene realization identity is not SHA-256")
    selected = tuple(report for report in reports if report.get("scene_id") == scene_id)
    if len(selected) != 1:
        raise SceneResultBindingError("scene execution report does not resolve exactly once")
    report = selected[0]
    evidence_fingerprint = report.get("execution_evidence_fingerprint")
    if (report.get("mode") != "execution" or report.get("errors") != [] or
            not _sha(evidence_fingerprint)):
        raise SceneResultBindingError("scene execution report is not execution-ready")
    receipts = report.get("evidence_receipts")
    if not isinstance(receipts, list):
        raise SceneResultBindingError("scene execution report has no receipt list")
    simulator = tuple(
        item for item in receipts
        if isinstance(item, Mapping) and item.get("check") == "simulator_integrity"
    )
    if len(simulator) != 1:
        raise SceneResultBindingError("simulator-integrity receipt does not resolve exactly once")
    artifacts = simulator[0].get("raw_artifacts")
    if not isinstance(artifacts, list):
        raise SceneResultBindingError("simulator-integrity receipt has no raw artifacts")
    compiled = tuple(
        item for item in artifacts
        if isinstance(item, Mapping) and item.get("artifact_type") == "compiled_model"
    )
    if len(compiled) != 1 or not _sha(compiled[0].get("sha256")):
        raise SceneResultBindingError("compiled-model artifact does not resolve exactly once")
    return SceneResultBinding(
        scene_fingerprint=scene_fingerprint,
        scene_realization_sha256=scene_realization_sha256,
        scene_execution_evidence_fingerprint=str(evidence_fingerprint),
        compiled_model_sha256=str(compiled[0]["sha256"]),
    )


def _sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


__all__ = ["SceneResultBinding", "SceneResultBindingError", "resolve_scene_result_binding"]
