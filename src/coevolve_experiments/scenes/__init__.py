"""Hashed scene registry and evidence-aware preflight."""

from .evidence import (
    ALLOWED_EVIDENCE_PRODUCERS,
    ALLOWED_SCRIPTED_SOLVABILITY_CONTROLLERS,
    EVIDENCE_CHECKS,
    EvidenceReceiptBinding,
    FrozenControllerPin,
    RawArtifactBinding,
    execution_evidence_fingerprint,
    validate_evidence_receipt,
)
from .integrity import ScenePreflightError, SceneValidationReport, validate_scene
from .evidence_producer import (
    EvidenceProductionBlocked,
    EvidenceProductionError,
    SimulatorIntegrityResult,
    produce_scene_evidence,
    produce_simulator_integrity_evidence,
)
from .loader import (
    compute_asset_composite_hash,
    compute_scene_hash,
    directed_route_document,
    load_scene,
    scene_hash_from_mapping,
)
from .materializer import (
    MATERIALIZATION_RECEIPT_KIND,
    MATERIALIZATION_RECEIPT_SCHEMA_VERSION,
    MATERIALIZER_ID,
    MATERIALIZER_VERSION,
    MaterializationResult,
    MaterializedTransform,
    SceneMaterializationError,
    inspect_materialized_transform,
    materialize_scene,
)
from .schema import SceneSpec

__all__ = [
    "ALLOWED_EVIDENCE_PRODUCERS",
    "ALLOWED_SCRIPTED_SOLVABILITY_CONTROLLERS",
    "EVIDENCE_CHECKS",
    "EvidenceReceiptBinding",
    "EvidenceProductionBlocked",
    "EvidenceProductionError",
    "FrozenControllerPin",
    "MATERIALIZATION_RECEIPT_KIND",
    "MATERIALIZATION_RECEIPT_SCHEMA_VERSION",
    "MATERIALIZER_ID",
    "MATERIALIZER_VERSION",
    "MaterializationResult",
    "MaterializedTransform",
    "RawArtifactBinding",
    "ScenePreflightError",
    "SceneMaterializationError",
    "SceneSpec",
    "SceneValidationReport",
    "SimulatorIntegrityResult",
    "compute_asset_composite_hash",
    "compute_scene_hash",
    "directed_route_document",
    "execution_evidence_fingerprint",
    "inspect_materialized_transform",
    "load_scene",
    "materialize_scene",
    "produce_scene_evidence",
    "produce_simulator_integrity_evidence",
    "scene_hash_from_mapping",
    "validate_evidence_receipt",
    "validate_scene",
]
