"""Fail-closed construction of the canonical Tables 2--4 protocol lock.

The low-level :mod:`coevolve_experiments.lockfile` module deliberately only
parses an already frozen lock.  This module owns the more demanding operation
of *creating* that lock from evidence.  In particular, it does not accept a
caller-authored commit, provider route, source identity, or digest:

* both Table-1 cohorts are reconstructed by the literal V83 source auditor;
* both Table-4 banks are recomputed from five audited source runs and their
  verifier-evidence packages;
* the recomputed banks must reproduce an existing candidate bank,
  provenance document, and candidate attestation; and
* protocol-contract and method-prompt fingerprints are derived from real,
  byte-addressed artifacts.

The resulting protocol ID is a domain-separated SHA-256 identity over the
complete registered paper contract, declarative matrix, scene registry,
run/output layout, evidence-derived lock material, and candidate attestations.
No credential value or filesystem path is serialized into the lock identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .contracts import (
    MethodDecisionPromptPin,
    MethodVariant,
    ProtocolLock,
    Table1SourceCohort,
)
from .lockfile import LOCK_SCHEMA_VERSION, protocol_lock_document
from .memory import (
    export_audited_table1_strategy_bank,
    load_posthoc_source_evidence,
    load_strategy_bank_candidate_attestation,
    validate_locked_strategy_bank_candidate,
    validate_strategy_bank,
)
from .paper_contract import build_paper_contract
from .protocols import load_protocol_bundle
from .provider_routes import PRIMARY_GPT_PROVIDER_ROUTE
from .registry import build_suite
from .reporting import audit_frozen_table1_candidate
from .scenes import load_scene
from .serialization import sha256_json, strict_json_loads, to_primitive


PROTOCOL_CONTRACT_ARTIFACT_KIND = "coevolve_protocol_contract_artifact"
PROTOCOL_CONTRACT_ARTIFACT_SCHEMA_VERSION = 1
PROTOCOL_LOCK_IDENTITY_KIND = "coevolve_tables_2_4_protocol_identity"
PROTOCOL_LOCK_IDENTITY_SCHEMA_VERSION = 1
PROTOCOL_LOCK_BUILD_RECEIPT_KIND = "coevolve_protocol_lock_build_receipt"
PROTOCOL_LOCK_BUILD_RECEIPT_SCHEMA_VERSION = 1
PROTOCOL_LOCK_BUILD_INPUT_KIND = "coevolve_protocol_lock_build_inputs"
PROTOCOL_LOCK_BUILD_INPUT_SCHEMA_VERSION = 1
PROTOCOL_ID_PREFIX = "coevolve-tables-2-4-v1"

PROTOCOL_ARTIFACT_ROLES = (
    "invariant",
    "terminal_predicate",
    "observation_contract",
    "task_instruction",
)
SHARED_PROMPT_ARTIFACT_ROLE = "shared_prompt_contract"

_ARTIFACT_FIELDS = frozenset(
    {"kind", "schema_version", "role", "payload", "payload_sha256"}
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bBearer\s+[^\s\"']+", re.IGNORECASE),
)
_SECRET_FIELD_NAMES = frozenset(
    {"api_key", "apikey", "authorization", "bearer_token", "secret", "token"}
)


class ProtocolLockBuildError(RuntimeError):
    """The evidence set is not sufficient to freeze a canonical lock."""


@dataclass(frozen=True, slots=True)
class ProtocolContractArtifactPin:
    """Path-free identity of one reviewed protocol-contract artifact."""

    role: str
    sha256: str
    payload_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ProtocolContractArtifacts:
    """Existing canonical artifact files which define the shared contract."""

    invariant: Path
    terminal_predicate: Path
    observation_contract: Path
    task_instruction: Path
    shared_prompt_contract: Path

    def paths_by_role(self) -> Mapping[str, Path]:
        return {
            role: Path(getattr(self, role)) for role in PROTOCOL_ARTIFACT_ROLES
        }


@dataclass(frozen=True, slots=True)
class StrategySourceInputs:
    """Complete pre-lock evidence for one Table-4 source backbone."""

    source_manifest: Path
    source_runs: tuple[Path, ...]
    evidence_package: Path
    candidate_bank: Path
    candidate_provenance: Path
    candidate_attestation: Path
    parent_manifest: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_runs, tuple) or len(self.source_runs) != 5:
            raise ProtocolLockBuildError(
                "strategy source inputs require exactly five source-run paths"
            )
        if len(set(self.source_runs)) != 5:
            raise ProtocolLockBuildError("strategy source-run paths must be unique")


@dataclass(frozen=True, slots=True)
class ProtocolLockBuild:
    """An in-memory lock plus its path-free canonical identity receipt."""

    lock: ProtocolLock
    identity: Mapping[str, Any]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ProtocolLockBuildInputs:
    """Strict manifest-decoded inputs for one evidence-derived lock build."""

    gpt_source: StrategySourceInputs
    gemini_source: StrategySourceInputs
    contract_artifacts: ProtocolContractArtifacts


@dataclass(frozen=True, slots=True)
class PersistedProtocolLockBuild:
    """Locations and exact byte hashes of an idempotently persisted build."""

    lock_path: Path
    lock_file_sha256: str
    identity_path: Path
    identity_file_sha256: str
    protocol_id: str


def protocol_contract_artifact_document(
    role: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a canonical, self-hashed contract artifact document.

    This helper does not write anything and does not imply that the payload has
    received scientific review.  The lock builder only accepts the resulting
    strict shape and additionally refuses credential-like material.
    """

    if role not in PROTOCOL_ARTIFACT_ROLES:
        raise ProtocolLockBuildError(f"unknown protocol artifact role: {role!r}")
    if not isinstance(payload, Mapping) or not payload:
        raise ProtocolLockBuildError("protocol artifact payload must be non-empty")
    primitive = to_primitive(payload)
    _reject_credentials(primitive, f"{role} protocol artifact")
    return {
        "kind": PROTOCOL_CONTRACT_ARTIFACT_KIND,
        "schema_version": PROTOCOL_CONTRACT_ARTIFACT_SCHEMA_VERSION,
        "role": role,
        "payload": primitive,
        "payload_sha256": sha256_json(primitive),
    }


def protocol_lock_build_input_document(
    *,
    gpt_source: StrategySourceInputs,
    gemini_source: StrategySourceInputs,
    contract_artifacts: ProtocolContractArtifacts,
) -> dict[str, Any]:
    """Serialize only paths and no credentials into a self-hashed input file."""

    payload: dict[str, Any] = {
        "kind": PROTOCOL_LOCK_BUILD_INPUT_KIND,
        "schema_version": PROTOCOL_LOCK_BUILD_INPUT_SCHEMA_VERSION,
        "gpt_source": _strategy_source_input_document(gpt_source),
        "gemini_source": _strategy_source_input_document(gemini_source),
        "contract_artifacts": {
            "invariant": str(contract_artifacts.invariant),
            "terminal_predicate": str(contract_artifacts.terminal_predicate),
            "observation_contract": str(contract_artifacts.observation_contract),
            "task_instruction": str(contract_artifacts.task_instruction),
            "shared_prompt_contract": str(
                contract_artifacts.shared_prompt_contract
            ),
        },
    }
    _reject_credentials(payload, "protocol-lock build input")
    payload["input_payload_sha256"] = sha256_json(payload)
    return payload


def load_protocol_lock_build_inputs(
    path: str | Path,
) -> ProtocolLockBuildInputs:
    """Load a canonical absolute-path manifest and validate every artifact."""

    _, raw = _read_regular_file(path, "protocol-lock build input manifest")
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolLockBuildError(
            f"protocol-lock build input manifest is not strict JSON: {exc}"
        ) from exc
    required = {
        "kind",
        "schema_version",
        "gpt_source",
        "gemini_source",
        "contract_artifacts",
        "input_payload_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ProtocolLockBuildError(
            "protocol-lock build input fields differ: "
            f"missing={sorted(required - actual)}, unknown={sorted(actual - required)}"
        )
    if value.get("kind") != PROTOCOL_LOCK_BUILD_INPUT_KIND:
        raise ProtocolLockBuildError("protocol-lock build input kind is invalid")
    if value.get("schema_version") != PROTOCOL_LOCK_BUILD_INPUT_SCHEMA_VERSION:
        raise ProtocolLockBuildError("protocol-lock build input schema is invalid")
    unsigned = dict(value)
    declared_hash = unsigned.pop("input_payload_sha256")
    if declared_hash != sha256_json(unsigned):
        raise ProtocolLockBuildError("protocol-lock build input self-hash mismatch")
    _reject_credentials(value, "protocol-lock build input")
    if raw != _render_json(value):
        raise ProtocolLockBuildError(
            "protocol-lock build input bytes are not canonical JSON"
        )
    gpt = _strategy_source_inputs_from_mapping(value["gpt_source"], "gpt")
    gemini = _strategy_source_inputs_from_mapping(
        value["gemini_source"], "gemini"
    )
    artifact_value = value["contract_artifacts"]
    artifact_fields = {
        "invariant",
        "terminal_predicate",
        "observation_contract",
        "task_instruction",
        "shared_prompt_contract",
    }
    if not isinstance(artifact_value, Mapping) or set(artifact_value) != artifact_fields:
        raise ProtocolLockBuildError(
            "protocol-lock build contract_artifacts fields are invalid"
        )
    if not all(isinstance(artifact_value[field], str) for field in artifact_fields):
        raise ProtocolLockBuildError("contract artifact paths must be strings")
    artifacts = ProtocolContractArtifacts(
        **{field: Path(artifact_value[field]) for field in artifact_fields}
    )
    if gpt.parent_manifest is not None:
        raise ProtocolLockBuildError("GPT build input parent_manifest must be null")
    if gemini.parent_manifest is None:
        raise ProtocolLockBuildError(
            "Gemini build input must bind the exact GPT parent manifest"
        )
    _require_same_file(
        gemini.parent_manifest,
        gpt.source_manifest,
        "Gemini parent manifest and GPT source manifest",
    )
    _validate_strategy_source_paths("gpt", gpt)
    _validate_strategy_source_paths("gemini", gemini)
    _load_contract_artifacts(artifacts)
    return ProtocolLockBuildInputs(gpt, gemini, artifacts)


def canonical_protocol_contract_payloads() -> dict[str, dict[str, Any]]:
    """Return the exact reviewed shared-contract payloads for Tables 2--4.

    Keeping these values in code as well as in byte-addressed artifacts makes
    the builder check scientific meaning, not merely the existence of four
    arbitrary files.  The values mirror Sections 3--4 of the manuscript and
    the frozen V83 formal protocol.
    """

    return {
        "invariant": {
            "physical_transaction_budget": 80,
            "seeds": [200, 201, 202, 203, 204],
            "foundation_model_control_frequency": (
                "between_physical_transactions_only"
            ),
            "team_dispatch": "single_atomic_front_back_barrier",
            "rollback_checkpoint": "latest_accepted",
            "exact_rejected_transaction_scope": "same_trusted_checkpoint",
            "no_human_intervention": True,
            "common_across_conditions": [
                "observation_contract",
                "task_instruction",
                "sonic_skill_api",
                "physics_verifier",
                "checkpoint_rollback",
                "terminal_predicate",
                "physical_budget",
            ],
            "valid_terminals": [
                "SUCCESS",
                "SYSTEM_STALL",
                "PHYSICAL_BUDGET_EXHAUSTED",
                "CONSECUTIVE_PAYLOAD_DROP_LIMIT",
            ],
            "infrastructure_invalid_terminals": [
                "PROVIDER_FATAL",
                "BACKEND_FATAL",
                "HOST_CONTRACT_ERROR",
            ],
            "system_stall_recovery_cycles": 3,
            "consecutive_payload_drop_limit": {
                "count": 5,
                "scope": "same_trusted_checkpoint",
                "reset_on": [
                    "accepted_transition",
                    "non_drop_physical_outcome",
                    "checkpoint_change",
                ],
            },
        },
        "terminal_predicate": {
            "payload_com_within_target_support_radius_m": 0.30,
            "payload_center_z_error_max_m": 0.08,
            "payload_tilt_max_deg": 20.0,
            "stable_duration_s": 3.0,
            "both_robots_upright": True,
            "post_placement_retreat_required": True,
            "robot_target_clearance_min_m": 1.10,
            "reported_alignment_xy_m": 0.18,
            "reported_alignment_is_success_gate": False,
            "authority": "mujoco_state_contacts_forces",
        },
        "observation_contract": {
            "inputs": [
                "natural_language_task_instruction",
                "compact_physical_state_serialization",
                "fixed_global_camera_view",
                "payload_centered_camera_view",
                "condition_permitted_cooperative_experience",
                "host_validated_directed_route_reference",
            ],
            "route_reference_fields": [
                "connected_route_geometry",
                "local_direction",
                "progress_gates",
            ],
            "route_reference_must_not_prescribe": [
                "coordination_primitive",
                "coordination_mode",
                "temporary_inner_outer_assignment",
                "bounded_action_parameters",
            ],
            "physical_acceptance_authority": "host_physics_verifier",
        },
        "task_instruction": {
            "task_id": "two_g1_long_box_carry_manual_layout",
            "instruction": (
                "Two G1 robots carry the long_box through the authored scene "
                "and stably place it on the red target cylinder."
            ),
            "lifecycle": [
                "joint_transport",
                "multi_segment_corridor_navigation",
                "placement",
                "release",
                "retreat",
                "upright_recovery",
            ],
        },
    }


def build_canonical_protocol_lock(
    *,
    gpt_source: StrategySourceInputs,
    gemini_source: StrategySourceInputs,
    contract_artifacts: ProtocolContractArtifacts,
    protocol_root: Path | None = None,
    scene_root: Path | None = None,
) -> ProtocolLockBuild:
    """Construct a protocol lock only after all source evidence is complete.

    The function is read-only.  It performs no simulator/model call, does not
    create a result directory, and never mutates a Table-1 run.  Call
    :func:`write_protocol_lock_build` only after this function returns.
    """

    if not isinstance(gpt_source, StrategySourceInputs) or not isinstance(
        gemini_source, StrategySourceInputs
    ):
        raise ProtocolLockBuildError(
            "gpt_source and gemini_source must be StrategySourceInputs"
        )
    if not isinstance(contract_artifacts, ProtocolContractArtifacts):
        raise ProtocolLockBuildError(
            "contract_artifacts must be ProtocolContractArtifacts"
        )
    if gpt_source.parent_manifest is not None:
        raise ProtocolLockBuildError("GPT source must not declare a parent manifest")
    if gemini_source.parent_manifest is None:
        raise ProtocolLockBuildError(
            "Gemini source requires the exact audited-V83 GPT parent manifest"
        )
    _require_same_file(
        gemini_source.parent_manifest,
        gpt_source.source_manifest,
        "Gemini parent manifest and GPT source manifest",
    )
    _validate_strategy_source_paths("gpt", gpt_source)
    _validate_strategy_source_paths("gemini", gemini_source)

    artifact_pins = _load_contract_artifacts(contract_artifacts)

    try:
        gpt_audit = audit_frozen_table1_candidate(
            gpt_source.source_manifest,
            "gpt",
        )
        gemini_audit = audit_frozen_table1_candidate(
            gemini_source.source_manifest,
            "gemini",
            parent_manifest_path=gemini_source.parent_manifest,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProtocolLockBuildError(
            f"Table-1 source cohort audit failed: {exc}"
        ) from exc
    for label, audit in (("GPT", gpt_audit), ("Gemini", gemini_audit)):
        if getattr(audit, "protocol_valid", False) is not True:
            raise ProtocolLockBuildError(
                f"{label} Table-1 source is not a complete audited five-run cohort"
            )

    full_method = gpt_audit.full_method_compatibility
    if gpt_audit.source.full_method_compatibility_fingerprint != full_method.fingerprint:
        raise ProtocolLockBuildError(
            "GPT source and audited Full-method compatibility identities differ"
        )
    if (
        gpt_audit.source.shared_prompt_contract_fingerprint
        != full_method.prompt_sha256
        or gpt_audit.source.decision_prompt_fingerprint
        != full_method.prompt_sha256
    ):
        raise ProtocolLockBuildError(
            "audited V83 prompt bytes do not bind both shared and Full prompt pins"
        )
    if (
        gemini_audit.source.shared_prompt_contract_fingerprint
        != gpt_audit.source.shared_prompt_contract_fingerprint
        or gemini_audit.source.decision_prompt_fingerprint
        != gpt_audit.source.decision_prompt_fingerprint
    ):
        raise ProtocolLockBuildError(
            "Gemini provider companion does not preserve the audited V83 prompt"
        )
    artifact_by_role = {pin.role: pin for pin in artifact_pins}
    if (
        artifact_by_role[SHARED_PROMPT_ARTIFACT_ROLE].sha256
        != full_method.prompt_sha256
    ):
        raise ProtocolLockBuildError(
            "shared prompt artifact bytes differ from the audited V83 prompt"
        )

    gpt_candidate = _audit_strategy_candidate("gpt", gpt_source, gpt_audit)
    gemini_candidate = _audit_strategy_candidate(
        "gemini", gemini_source, gemini_audit
    )
    prompt_pins, prompt_registry_fingerprint = _build_method_prompt_identity(
        full_method.prompt_sha256
    )

    lock_material: dict[str, Any] = {
        "invariant_fingerprint": artifact_by_role["invariant"].sha256,
        "terminal_fingerprint": artifact_by_role["terminal_predicate"].sha256,
        "base_method_commit": gpt_audit.source.method_commit,
        "table1_gpt_source": gpt_audit.source,
        "table1_gemini_source": gemini_audit.source,
        "gpt_strategy_bank_hash": gpt_candidate["bank_sha256"],
        "gemini_strategy_bank_hash": gemini_candidate["bank_sha256"],
        "target_provider_route_id": PRIMARY_GPT_PROVIDER_ROUTE.route_id,
        "target_provider_route_fingerprint": PRIMARY_GPT_PROVIDER_ROUTE.fingerprint,
        "observation_contract_fingerprint": artifact_by_role[
            "observation_contract"
        ].sha256,
        "task_instruction_fingerprint": artifact_by_role[
            "task_instruction"
        ].sha256,
        "shared_prompt_contract_fingerprint": full_method.prompt_sha256,
        "method_decision_prompt_pins": prompt_pins,
        "skill_api_fingerprint": full_method.skill_api_sha256,
        "full_method_compatibility": full_method,
    }

    try:
        bundle = load_protocol_bundle(protocol_root)
        paper_contract = build_paper_contract("full", protocol_root=protocol_root)
        scenes = {
            scene_id.value: load_scene(scene_id, root=scene_root)
            for scene_id in sorted(
                {condition.scene_id for table in bundle.tables for condition in table.conditions},
                key=lambda value: value.value,
            )
        }
    except (OSError, TypeError, ValueError) as exc:
        raise ProtocolLockBuildError(
            f"registered paper protocol or scene registry is invalid: {exc}"
        ) from exc

    run_layout = _declared_run_output_layout(bundle)
    identity = {
        "kind": PROTOCOL_LOCK_IDENTITY_KIND,
        "schema_version": PROTOCOL_LOCK_IDENTITY_SCHEMA_VERSION,
        "lock_schema_version": LOCK_SCHEMA_VERSION,
        "suite_tier": "full",
        "lock_material_sha256": sha256_json(lock_material),
        "paper_contract_sha256": sha256_json(paper_contract),
        "protocol_bundle_sha256": sha256_json(bundle),
        "scene_registry_sha256": sha256_json(scenes),
        "run_output_layout_sha256": sha256_json(run_layout),
        "contract_artifacts": to_primitive(artifact_pins),
        "decision_prompt_registry_sha256": prompt_registry_fingerprint,
        "strategy_candidates": {
            "gpt": gpt_candidate,
            "gemini": gemini_candidate,
        },
    }
    protocol_fingerprint = sha256_json(identity)
    protocol_id = f"{PROTOCOL_ID_PREFIX}-{protocol_fingerprint}"

    try:
        lock = ProtocolLock(protocol_id=protocol_id, **lock_material)
        suite = build_suite(
            "full",
            lock,
            protocol_root=protocol_root,
            scene_root=scene_root,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolLockBuildError(
            f"evidence-derived protocol lock does not compile: {exc}"
        ) from exc
    actual_layout = _compiled_run_output_layout(suite, bundle)
    if actual_layout != run_layout:
        raise ProtocolLockBuildError(
            "compiled run/output bindings differ from the identity preimage"
        )

    lock_document = protocol_lock_document(lock)
    lock_bytes = _render_json(lock_document)
    receipt: dict[str, Any] = {
        "kind": PROTOCOL_LOCK_BUILD_RECEIPT_KIND,
        "schema_version": PROTOCOL_LOCK_BUILD_RECEIPT_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "protocol_fingerprint": protocol_fingerprint,
        "identity": identity,
        "protocol_lock_semantic_sha256": sha256_json(lock_document),
        "protocol_lock_file_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "run_output_bindings_sha256": sha256_json(actual_layout),
        "new_run_count": suite.unique_run_count,
        "reuse_count": len(suite.reuses),
        "paper_evidence": False,
        "physics_dispatched": False,
        "execution_ready": False,
    }
    receipt["receipt_payload_sha256"] = sha256_json(receipt)
    _reject_credentials(receipt, "protocol-lock build receipt")
    return ProtocolLockBuild(lock=lock, identity=identity, receipt=receipt)


def write_protocol_lock_build(
    build: ProtocolLockBuild,
    lock_path: str | Path,
    identity_path: str | Path | None = None,
) -> PersistedProtocolLockBuild:
    """Persist a lock and identity receipt without overwriting different bytes."""

    if not isinstance(build, ProtocolLockBuild):
        raise ProtocolLockBuildError("build must be a ProtocolLockBuild")
    destination = _absolute_output_path(lock_path, "protocol lock output")
    receipt_destination = _absolute_output_path(
        identity_path
        if identity_path is not None
        else destination.with_name(f"{destination.stem}.identity.json"),
        "protocol identity output",
    )
    if destination == receipt_destination:
        raise ProtocolLockBuildError("lock and identity outputs must be different files")

    lock_bytes = _render_json(protocol_lock_document(build.lock))
    receipt_bytes = _render_json(build.receipt)
    expected_lock_sha = str(build.receipt.get("protocol_lock_file_sha256", ""))
    if hashlib.sha256(lock_bytes).hexdigest() != expected_lock_sha:
        raise ProtocolLockBuildError("in-memory lock differs from its build receipt")

    _preflight_idempotent_output(destination, lock_bytes, "protocol lock")
    _preflight_idempotent_output(
        receipt_destination, receipt_bytes, "protocol identity receipt"
    )
    _write_new_or_identical(destination, lock_bytes)
    _write_new_or_identical(receipt_destination, receipt_bytes)
    return PersistedProtocolLockBuild(
        lock_path=destination,
        lock_file_sha256=hashlib.sha256(lock_bytes).hexdigest(),
        identity_path=receipt_destination,
        identity_file_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        protocol_id=build.lock.protocol_id,
    )


def _load_contract_artifacts(
    artifacts: ProtocolContractArtifacts,
) -> tuple[ProtocolContractArtifactPin, ...]:
    paths = artifacts.paths_by_role()
    canonical_payloads = canonical_protocol_contract_payloads()
    resolved_paths: list[Path] = []
    pins: list[ProtocolContractArtifactPin] = []
    for role in PROTOCOL_ARTIFACT_ROLES:
        path, raw = _read_regular_file(paths[role], f"{role} protocol artifact")
        resolved_paths.append(path)
        try:
            document = strict_json_loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProtocolLockBuildError(
                f"{role} protocol artifact is not strict JSON: {exc}"
            ) from exc
        if not isinstance(document, Mapping) or set(document) != _ARTIFACT_FIELDS:
            actual = set(document) if isinstance(document, Mapping) else set()
            raise ProtocolLockBuildError(
                f"{role} protocol artifact fields differ: "
                f"missing={sorted(_ARTIFACT_FIELDS - actual)}, "
                f"unknown={sorted(actual - _ARTIFACT_FIELDS)}"
            )
        if document.get("kind") != PROTOCOL_CONTRACT_ARTIFACT_KIND:
            raise ProtocolLockBuildError(f"{role} protocol artifact kind is invalid")
        if (
            document.get("schema_version")
            != PROTOCOL_CONTRACT_ARTIFACT_SCHEMA_VERSION
        ):
            raise ProtocolLockBuildError(f"{role} protocol artifact schema is invalid")
        if document.get("role") != role:
            raise ProtocolLockBuildError(
                f"{role} protocol artifact declares the wrong role"
            )
        payload = document.get("payload")
        if not isinstance(payload, Mapping) or not payload:
            raise ProtocolLockBuildError(
                f"{role} protocol artifact payload must be a non-empty object"
            )
        payload_sha = document.get("payload_sha256")
        if payload_sha != sha256_json(payload):
            raise ProtocolLockBuildError(
                f"{role} protocol artifact payload hash mismatch"
            )
        if payload != canonical_payloads[role]:
            raise ProtocolLockBuildError(
                f"{role} protocol artifact semantics differ from the registered "
                "Tables 2--4 contract"
            )
        _reject_credentials(document, f"{role} protocol artifact")
        if raw != _render_json(document):
            raise ProtocolLockBuildError(
                f"{role} protocol artifact bytes are not canonical JSON"
            )
        pins.append(
            ProtocolContractArtifactPin(
                role=role,
                sha256=hashlib.sha256(raw).hexdigest(),
                payload_sha256=str(payload_sha),
                size_bytes=len(raw),
            )
        )
    prompt_path, prompt_raw = _read_regular_file(
        artifacts.shared_prompt_contract,
        "shared prompt contract artifact",
    )
    resolved_paths.append(prompt_path)
    try:
        prompt_text = prompt_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolLockBuildError(
            "shared prompt contract artifact must be UTF-8 text"
        ) from exc
    _reject_credentials(prompt_text, "shared prompt contract artifact")
    prompt_sha256 = hashlib.sha256(prompt_raw).hexdigest()
    pins.append(
        ProtocolContractArtifactPin(
            role=SHARED_PROMPT_ARTIFACT_ROLE,
            sha256=prompt_sha256,
            payload_sha256=prompt_sha256,
            size_bytes=len(prompt_raw),
        )
    )
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ProtocolLockBuildError(
            "each protocol artifact role must use a distinct existing file"
        )
    return tuple(pins)


def _audit_strategy_candidate(
    cohort: str,
    source: StrategySourceInputs,
    cohort_audit: Any,
) -> dict[str, Any]:
    """Recompute one candidate chain; no attestation-only shortcut exists."""

    for index, source_run in enumerate(source.source_runs):
        _require_absolute_directory(source_run, f"{cohort} source run {index}")
    for path, label in (
        (source.candidate_bank, f"{cohort} candidate bank"),
        (source.candidate_provenance, f"{cohort} candidate provenance"),
        (source.candidate_attestation, f"{cohort} candidate attestation"),
    ):
        _read_regular_file(path, label)
    try:
        package = load_posthoc_source_evidence(
            source.evidence_package,
            cohort_audit,
        )
        recomputed = export_audited_table1_strategy_bank(
            source.source_runs,
            cohort_audit,
            posthoc_evidence_package=source.evidence_package,
        )
        bank_sha256 = str(recomputed.provenance.get("bank_sha256", ""))
        candidate = validate_locked_strategy_bank_candidate(
            recomputed,
            cohort_audit,
            package,
            cohort,
            source.candidate_attestation,
            locked_bank_sha256=bank_sha256,
        )
        loaded_candidate = load_strategy_bank_candidate_attestation(
            source.candidate_attestation
        )
        bank_validation = validate_strategy_bank(source.candidate_bank)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProtocolLockBuildError(
            f"{cohort} strategy-bank candidate audit failed: {exc}"
        ) from exc
    if candidate.payload != loaded_candidate.payload:
        raise ProtocolLockBuildError(
            f"{cohort} candidate attestation changed during validation"
        )
    if bank_validation.bank_sha256 != bank_sha256:
        raise ProtocolLockBuildError(
            f"{cohort} candidate bank hash differs from the recomputed bank"
        )
    _require_json_matches(
        source.candidate_bank,
        recomputed.bank,
        f"{cohort} candidate bank",
    )
    _require_json_matches(
        source.candidate_provenance,
        recomputed.provenance,
        f"{cohort} candidate provenance",
    )
    if candidate.bank_sha256 != bank_sha256:
        raise ProtocolLockBuildError(
            f"{cohort} candidate attestation bank hash differs"
        )
    return {
        "source_cohort": cohort,
        "source": to_primitive(cohort_audit.source),
        "source_aggregate_manifest_sha256": cohort_audit.manifest_sha256,
        "posthoc_evidence_manifest_sha256": package.manifest_sha256,
        "posthoc_evidence_payload_sha256": package.payload_sha256,
        "bank_sha256": bank_sha256,
        "provenance_sha256": sha256_json(recomputed.provenance),
        "candidate_payload_sha256": candidate.payload[
            "candidate_payload_sha256"
        ],
        "candidate_file_sha256": candidate.file_sha256,
    }


def _validate_strategy_source_paths(
    cohort: str,
    source: StrategySourceInputs,
) -> None:
    _read_regular_file(source.source_manifest, f"{cohort} source manifest")
    if source.parent_manifest is not None:
        _read_regular_file(source.parent_manifest, f"{cohort} parent manifest")
    for index, source_run in enumerate(source.source_runs):
        _require_absolute_directory(source_run, f"{cohort} source run {index}")
    _require_absolute_directory(
        source.evidence_package,
        f"{cohort} post-hoc evidence package",
    )
    for path, label in (
        (source.candidate_bank, f"{cohort} candidate bank"),
        (source.candidate_provenance, f"{cohort} candidate provenance"),
        (source.candidate_attestation, f"{cohort} candidate attestation"),
    ):
        _read_regular_file(path, label)


def _strategy_source_input_document(source: StrategySourceInputs) -> dict[str, Any]:
    if not isinstance(source, StrategySourceInputs):
        raise ProtocolLockBuildError(
            "strategy source input document requires StrategySourceInputs"
        )
    return {
        "source_manifest": str(source.source_manifest),
        "source_runs": [str(path) for path in source.source_runs],
        "evidence_package": str(source.evidence_package),
        "candidate_bank": str(source.candidate_bank),
        "candidate_provenance": str(source.candidate_provenance),
        "candidate_attestation": str(source.candidate_attestation),
        "parent_manifest": (
            str(source.parent_manifest)
            if source.parent_manifest is not None
            else None
        ),
    }


def _strategy_source_inputs_from_mapping(
    value: object,
    cohort: str,
) -> StrategySourceInputs:
    expected = {
        "source_manifest",
        "source_runs",
        "evidence_package",
        "candidate_bank",
        "candidate_provenance",
        "candidate_attestation",
        "parent_manifest",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProtocolLockBuildError(
            f"{cohort} strategy source input fields are invalid"
        )
    scalar_fields = expected - {"source_runs", "parent_manifest"}
    if not all(isinstance(value[field], str) for field in scalar_fields):
        raise ProtocolLockBuildError(
            f"{cohort} strategy source artifact paths must be strings"
        )
    runs = value.get("source_runs")
    if not isinstance(runs, list) or not all(isinstance(path, str) for path in runs):
        raise ProtocolLockBuildError(
            f"{cohort} strategy source_runs must be a JSON string array"
        )
    parent = value.get("parent_manifest")
    if parent is not None and not isinstance(parent, str):
        raise ProtocolLockBuildError(
            f"{cohort} parent_manifest must be a path string or null"
        )
    return StrategySourceInputs(
        source_manifest=Path(value["source_manifest"]),
        source_runs=tuple(Path(path) for path in runs),
        evidence_package=Path(value["evidence_package"]),
        candidate_bank=Path(value["candidate_bank"]),
        candidate_provenance=Path(value["candidate_provenance"]),
        candidate_attestation=Path(value["candidate_attestation"]),
        parent_manifest=Path(parent) if parent is not None else None,
    )


def _build_method_prompt_identity(
    full_prompt_fingerprint: str,
) -> tuple[tuple[MethodDecisionPromptPin, ...], str]:
    """Load the checked-in prompt registry without creating an import cycle."""

    try:
        from .decision_prompts import build_production_prompt_identity

        identity = build_production_prompt_identity(
            shared_prompt_contract_fingerprint=full_prompt_fingerprint,
            full_method_prompt_fingerprint=full_prompt_fingerprint
        )
        pins = identity.method_decision_prompt_pins
        registry_fingerprint = identity.registry_fingerprint
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProtocolLockBuildError(
            f"production decision-prompt registry is unavailable: {exc}"
        ) from exc
    if not isinstance(pins, tuple) or not all(
        isinstance(pin, MethodDecisionPromptPin) for pin in pins
    ):
        raise ProtocolLockBuildError(
            "decision-prompt registry returned an invalid pin collection"
        )
    if tuple(pin.method_variant for pin in pins) != tuple(MethodVariant):
        raise ProtocolLockBuildError(
            "decision-prompt pins must follow the complete MethodVariant order"
        )
    if not isinstance(registry_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", registry_fingerprint
    ):
        raise ProtocolLockBuildError(
            "decision-prompt registry fingerprint is not a lowercase SHA-256"
        )
    if identity.shared_prompt_contract_fingerprint != full_prompt_fingerprint:
        raise ProtocolLockBuildError(
            "decision-prompt registry changed the shared V83 prompt fingerprint"
        )
    return pins, registry_fingerprint


def _declared_run_output_layout(bundle: Any) -> dict[str, Any]:
    new_runs: list[dict[str, Any]] = []
    reuses: list[dict[str, Any]] = []
    for table in bundle.tables:
        for condition in table.conditions:
            if condition.execution == "new":
                for seed in bundle.common.seeds:
                    new_runs.append(
                        {
                            "run_id": (
                                f"{table.table_id.value}__{condition.condition_id}__"
                                f"seed_{seed:03d}"
                            ),
                            "table_id": table.table_id.value,
                            "condition_id": condition.condition_id,
                            "seed": seed,
                            "output_relpath": (
                                f"{table.table_id.value}/{condition.condition_id}/"
                                f"seed_{seed:03d}"
                            ),
                        }
                    )
            else:
                reuses.append(
                    {
                        "reuse_id": f"{table.table_id.value}__{condition.condition_id}",
                        "table_id": table.table_id.value,
                        "condition_id": condition.condition_id,
                        "source": condition.source,
                    }
                )
    return {"new_runs": new_runs, "reuses": reuses}


def _compiled_run_output_layout(suite: Any, bundle: Any) -> dict[str, Any]:
    condition_by_key: dict[tuple[str, str], str] = {}
    for table in bundle.tables:
        for condition in table.conditions:
            condition_by_key[(table.table_id.value, condition.condition_id)] = (
                condition.condition_id
            )
    new_runs: list[dict[str, Any]] = []
    for run in suite.new_runs:
        prefix = f"{run.table_id.value}__"
        suffix = f"__seed_{run.seed:03d}"
        if not run.run_id.startswith(prefix) or not run.run_id.endswith(suffix):
            raise ProtocolLockBuildError(
                f"compiled run ID does not encode its table/seed: {run.run_id}"
            )
        condition_id = run.run_id[len(prefix) : -len(suffix)]
        if (run.table_id.value, condition_id) not in condition_by_key:
            raise ProtocolLockBuildError(
                f"compiled run ID has an unregistered condition: {run.run_id}"
            )
        new_runs.append(
            {
                "run_id": run.run_id,
                "table_id": run.table_id.value,
                "condition_id": condition_id,
                "seed": run.seed,
                "output_relpath": run.output_relpath,
            }
        )
    reuse_source_by_id = {
        f"{table.table_id.value}__{condition.condition_id}": condition.source
        for table in bundle.tables
        for condition in table.conditions
        if condition.execution == "reuse"
    }
    reuses = [
        {
            "reuse_id": reuse.reuse_id,
            "table_id": reuse.target_table.value,
            "condition_id": reuse.reuse_id.removeprefix(
                f"{reuse.target_table.value}__"
            ),
            "source": reuse_source_by_id.get(reuse.reuse_id),
        }
        for reuse in suite.reuses
    ]
    return {"new_runs": new_runs, "reuses": reuses}


def _require_json_matches(path: Path, expected: object, label: str) -> None:
    _, raw = _read_regular_file(path, label)
    try:
        actual = strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolLockBuildError(f"{label} is not strict JSON: {exc}") from exc
    if actual != to_primitive(expected):
        raise ProtocolLockBuildError(f"{label} differs from the audited recomputation")


def _read_regular_file(value: str | Path, label: str) -> tuple[Path, bytes]:
    path = Path(value)
    if not path.is_absolute():
        raise ProtocolLockBuildError(f"{label} path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ProtocolLockBuildError(f"{label} must be an existing non-symlink file")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ProtocolLockBuildError(f"{label} path must be canonical and symlink-free")
    raw = path.read_bytes()
    if not raw:
        raise ProtocolLockBuildError(f"{label} must not be empty")
    return resolved, raw


def _require_absolute_directory(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ProtocolLockBuildError(f"{label} path must be absolute")
    if path.is_symlink() or not path.is_dir():
        raise ProtocolLockBuildError(
            f"{label} must be an existing non-symlink directory"
        )
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ProtocolLockBuildError(f"{label} path must be canonical and symlink-free")
    return resolved


def _require_same_file(first: Path, second: Path, label: str) -> None:
    first_path, _ = _read_regular_file(first, label)
    second_path, _ = _read_regular_file(second, label)
    if first_path != second_path:
        raise ProtocolLockBuildError(f"{label} must identify the same exact file")


def _absolute_output_path(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ProtocolLockBuildError(f"{label} must be an absolute normalized path")
    if path.exists() and path.is_symlink():
        raise ProtocolLockBuildError(f"{label} must not be a symlink")
    resolved_parent = path.parent.resolve(strict=False)
    resolved = resolved_parent / path.name
    if resolved != path:
        raise ProtocolLockBuildError(f"{label} must be canonical and symlink-free")
    return path


def _preflight_idempotent_output(path: Path, content: bytes, label: str) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProtocolLockBuildError(f"{label} output is not a regular file")
        if path.read_bytes() != content:
            raise ProtocolLockBuildError(
                f"refusing to overwrite non-identical {label}: {path}"
            )


def _write_new_or_identical(path: Path, content: bytes) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ProtocolLockBuildError(
            f"protocol-lock temporary output already exists: {temporary}"
        )
    temporary.write_bytes(content)
    temporary.replace(path)


def _reject_credentials(value: object, label: str, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SECRET_FIELD_NAMES:
                raise ProtocolLockBuildError(
                    f"{label} contains forbidden credential field at {path}.{key}"
                )
            _reject_credentials(child, label, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_credentials(child, label, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS
    ):
        raise ProtocolLockBuildError(
            f"{label} contains credential-like material at {path}"
        )


def _render_json(value: object) -> bytes:
    return (
        json.dumps(
            to_primitive(value),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "PROTOCOL_ARTIFACT_ROLES",
    "PROTOCOL_CONTRACT_ARTIFACT_KIND",
    "PROTOCOL_CONTRACT_ARTIFACT_SCHEMA_VERSION",
    "PROTOCOL_ID_PREFIX",
    "PROTOCOL_LOCK_BUILD_RECEIPT_KIND",
    "PROTOCOL_LOCK_BUILD_RECEIPT_SCHEMA_VERSION",
    "PROTOCOL_LOCK_BUILD_INPUT_KIND",
    "PROTOCOL_LOCK_BUILD_INPUT_SCHEMA_VERSION",
    "PROTOCOL_LOCK_IDENTITY_KIND",
    "PROTOCOL_LOCK_IDENTITY_SCHEMA_VERSION",
    "SHARED_PROMPT_ARTIFACT_ROLE",
    "PersistedProtocolLockBuild",
    "ProtocolContractArtifactPin",
    "ProtocolContractArtifacts",
    "ProtocolLockBuild",
    "ProtocolLockBuildInputs",
    "ProtocolLockBuildError",
    "StrategySourceInputs",
    "build_canonical_protocol_lock",
    "canonical_protocol_contract_payloads",
    "load_protocol_lock_build_inputs",
    "protocol_contract_artifact_document",
    "protocol_lock_build_input_document",
    "write_protocol_lock_build",
]
