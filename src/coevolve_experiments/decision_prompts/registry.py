"""Strict loader and fingerprint registry for production decision prompts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Tuple

from ..contracts import MethodVariant
from ..serialization import sha256_json, strict_json_loads
from .contracts import (
    EXTERNAL_PROMPT_VARIANTS,
    FROZEN_V83_PROMPT_BINDING,
    PROMPT_ARTIFACT_KIND,
    PROMPT_ARTIFACT_SCHEMA_VERSION,
    PROMPT_FINGERPRINT_KIND,
    PROMPT_FINGERPRINT_SCHEMA_VERSION,
    PROMPT_REGISTRY_KIND,
    PROMPT_REGISTRY_SCHEMA_VERSION,
    PRODUCTION_ARTIFACT_SCOPE,
    PRODUCTION_POLICY_REPRESENTATION,
    SHARED_PROMPT_CONTRACT_BINDING,
    ProductionDecisionPromptArtifact,
    ProductionPromptArtifactError,
    ProductionPromptIdentity,
    ProductionPromptRegistry,
    ProductionPromptRole,
    PromptMemoryGates,
    _require_sha256,
)


_PACKAGE_ROOT = Path(__file__).resolve().parent
_REGISTRY_FILENAME = "registry.json"

_REGISTRY_FIELDS = frozenset({"kind", "schema_version", "artifacts"})
_REGISTRY_ENTRY_FIELDS = frozenset(
    {
        "method_variant",
        "manifest_relpath",
        "manifest_sha256",
        "decision_prompt_fingerprint",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "method_variant",
        "artifact_scope",
        "policy_representation",
        "shared_prompt_contract_binding",
        "frozen_v83_prompt_binding",
        "decision_structure",
        "hierarchical_evolution",
        "durable_dynamic_memory",
        "memory_gates",
        "common_runtime_roles",
        "forbidden_method_roles",
        "roles",
    }
)
_MEMORY_GATE_FIELDS = frozenset(
    {
        "strategy_retrieval",
        "strategy_promotion",
        "action_exemplar_retrieval",
        "action_exemplar_promotion",
        "memory_writer_call",
    }
)
_ROLE_FIELDS = frozenset(
    {"role", "prompt_base", "output_contract", "prompt_file"}
)
_COMMON_RUNTIME_ROLES = ("vlm_failure_analyzer", "code_repair")


@dataclass(frozen=True, slots=True)
class _ExpectedArtifact:
    decision_structure: str
    hierarchical_evolution: bool
    durable_dynamic_memory: bool
    memory_gates: PromptMemoryGates
    roles: Tuple[tuple[str, str, str], ...]
    forbidden_roles: Tuple[str, ...]


_DISABLED_GATES = PromptMemoryGates(False, False, False, False, False)
_ENABLED_GATES = PromptMemoryGates(True, True, True, True, True)
_EXPECTED: Mapping[MethodVariant, _ExpectedArtifact] = {
    MethodVariant.NO_DYNAMIC_EXPERIENCE: _ExpectedArtifact(
        decision_structure="hierarchical_evolution",
        hierarchical_evolution=True,
        durable_dynamic_memory=False,
        memory_gates=_DISABLED_GATES,
        roles=(
            (
                "task_orchestrator",
                "frozen_v83_task_orchestrator",
                "v83_task_orchestrator_json_v1",
            ),
            (
                "policy_generator",
                "frozen_v83_policy_coder",
                "v83_team_policy_json_v1",
            ),
            (
                "evolution_manager",
                "frozen_v83_evolution_manager",
                "v83_evolution_manager_json_v1",
            ),
            (
                "mutation_operator",
                "frozen_v83_mutation_operator",
                "v83_mutation_operator_json_v1",
            ),
        ),
        forbidden_roles=("memory_update",),
    ),
    MethodVariant.FLAT_REFLECTIVE_REPAIR: _ExpectedArtifact(
        decision_structure="flat_failure_conditioned_code_regeneration",
        hierarchical_evolution=False,
        durable_dynamic_memory=True,
        memory_gates=_ENABLED_GATES,
        roles=(
            (
                "flat_task_planner",
                "frozen_v83_task_orchestrator",
                "v83_task_orchestrator_json_v1",
            ),
            (
                "flat_strategy_selector",
                "frozen_v83_task_orchestrator",
                "v83_flat_strategy_commit_json_v1",
            ),
            (
                "flat_policy_generator",
                "frozen_v83_policy_coder",
                "v83_team_policy_json_v1",
            ),
            (
                "reflective_rewrite",
                "frozen_v83_policy_coder",
                "v83_team_policy_json_v1",
            ),
            (
                "memory_update",
                "frozen_v83_memory_update",
                "v83_memory_update_json_v1",
            ),
        ),
        forbidden_roles=("evolution_manager", "mutation_operator"),
    ),
    MethodVariant.REMAC_ADAPTED: _ExpectedArtifact(
        decision_structure="task_plan_reflection_and_replanning",
        hierarchical_evolution=False,
        durable_dynamic_memory=False,
        memory_gates=_DISABLED_GATES,
        roles=(
            ("remac_planner", "none", "remac_task_plan_json_v1"),
            (
                "remac_precondition_reflection",
                "frozen_v83_policy_coder",
                "v83_team_policy_json_v1",
            ),
            (
                "remac_postcondition_reflection",
                "none",
                "remac_postcondition_reflection_json_v1",
            ),
            ("remac_replanner", "none", "remac_task_plan_json_v1"),
        ),
        forbidden_roles=(
            "evolution_manager",
            "mutation_operator",
            "memory_update",
        ),
    ),
    MethodVariant.COHERENT_ADAPTED: _ExpectedArtifact(
        decision_structure="central_proposal_execution_feedback_adjustment",
        hierarchical_evolution=False,
        durable_dynamic_memory=False,
        memory_gates=_DISABLED_GATES,
        roles=(
            (
                "coherent_central_proposer",
                "frozen_v83_policy_coder",
                "v83_team_policy_json_v1",
            ),
            (
                "coherent_front_feasibility",
                "none",
                "coherent_executor_feasibility_json_v1",
            ),
            (
                "coherent_back_feasibility",
                "none",
                "coherent_executor_feasibility_json_v1",
            ),
            (
                "coherent_feedback_adjuster",
                "frozen_v83_policy_coder",
                "v83_team_policy_json_v1",
            ),
        ),
        forbidden_roles=(
            "evolution_manager",
            "mutation_operator",
            "memory_update",
        ),
    ),
}


def load_production_prompt_registry(
    root: str | Path | None = None,
) -> ProductionPromptRegistry:
    """Load all four artifacts and verify every byte before returning.

    ``root`` is the directory containing ``registry.json``.  It exists only
    for offline auditing/tests; formal callers should use the package-owned
    default and then bind the returned fingerprints into a protocol lock.
    """

    source_root = _PACKAGE_ROOT if root is None else Path(root)
    source_root = _require_directory(source_root, "prompt registry root")
    registry_path = _safe_regular_file(
        source_root,
        _REGISTRY_FILENAME,
        "prompt registry",
    )
    registry_bytes = _read_bytes(registry_path, "prompt registry")
    registry = _strict_object(registry_bytes, "prompt registry")
    _require_exact_fields(registry, _REGISTRY_FIELDS, "prompt registry")
    if registry["kind"] != PROMPT_REGISTRY_KIND:
        raise ProductionPromptArtifactError("prompt registry kind is invalid")
    if registry["schema_version"] != PROMPT_REGISTRY_SCHEMA_VERSION:
        raise ProductionPromptArtifactError(
            "prompt registry schema_version is unsupported"
        )
    entries = registry["artifacts"]
    if not isinstance(entries, list):
        raise ProductionPromptArtifactError(
            "prompt registry artifacts must be an array"
        )
    artifacts = tuple(
        _load_registry_entry(source_root, item, index)
        for index, item in enumerate(entries)
    )
    return ProductionPromptRegistry(
        artifacts=artifacts,
        registry_path=registry_path,
        registry_sha256=_sha256_bytes(registry_bytes),
    )


def build_production_prompt_identity(
    *,
    shared_prompt_contract_fingerprint: str,
    full_method_prompt_fingerprint: str,
    root: str | Path | None = None,
) -> ProductionPromptIdentity:
    """Build the exact prompt fields required by ``ProtocolLock``.

    Both caller-supplied values must come from real, independently audited
    artifacts.  This function neither discovers a convenient prompt nor
    manufactures a placeholder digest.
    """

    _require_sha256(
        "shared_prompt_contract_fingerprint",
        shared_prompt_contract_fingerprint,
    )
    registry = load_production_prompt_registry(root)
    return ProductionPromptIdentity(
        shared_prompt_contract_fingerprint=shared_prompt_contract_fingerprint,
        method_decision_prompt_pins=registry.method_decision_prompt_pins(
            full_method_prompt_fingerprint=full_method_prompt_fingerprint,
        ),
        registry_fingerprint=registry.registry_sha256,
    )


def decision_prompt_fingerprint_payload(
    *,
    method_variant: MethodVariant,
    manifest_sha256: str,
    prompt_file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Public, byte-recomputable fingerprint recipe used by auditors."""

    _require_sha256("manifest_sha256", manifest_sha256)
    if method_variant not in EXTERNAL_PROMPT_VARIANTS:
        raise ProductionPromptArtifactError(
            "fingerprint recipe is only defined for external prompt variants"
        )
    checked: dict[str, str] = {}
    for relpath, digest in sorted(prompt_file_hashes.items()):
        _require_safe_relpath(relpath, "prompt file")
        _require_sha256(f"prompt file {relpath}", digest)
        checked[relpath] = digest
    return {
        "kind": PROMPT_FINGERPRINT_KIND,
        "schema_version": PROMPT_FINGERPRINT_SCHEMA_VERSION,
        "method_variant": method_variant.value,
        "manifest_sha256": manifest_sha256,
        "prompt_file_sha256": checked,
    }


def _load_registry_entry(
    root: Path,
    value: object,
    index: int,
) -> ProductionDecisionPromptArtifact:
    label = f"prompt registry artifacts[{index}]"
    if not isinstance(value, Mapping):
        raise ProductionPromptArtifactError(f"{label} must be an object")
    entry = dict(value)
    _require_exact_fields(entry, _REGISTRY_ENTRY_FIELDS, label)
    try:
        variant = MethodVariant(entry["method_variant"])
    except (TypeError, ValueError) as exc:
        raise ProductionPromptArtifactError(
            f"{label}.method_variant is invalid"
        ) from exc
    if variant not in EXTERNAL_PROMPT_VARIANTS:
        raise ProductionPromptArtifactError(
            f"{label} cannot replace frozen Full CoEvolve"
        )
    manifest_relpath = entry["manifest_relpath"]
    if not isinstance(manifest_relpath, str):
        raise ProductionPromptArtifactError(
            f"{label}.manifest_relpath must be a string"
        )
    _require_sha256(f"{label}.manifest_sha256", entry["manifest_sha256"])
    _require_sha256(
        f"{label}.decision_prompt_fingerprint",
        entry["decision_prompt_fingerprint"],
    )
    manifest_path = _safe_regular_file(root, manifest_relpath, label)
    manifest_bytes = _read_bytes(manifest_path, label)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if manifest_sha256 != entry["manifest_sha256"]:
        raise ProductionPromptArtifactError(
            f"{label} manifest byte hash differs from registry"
        )
    manifest = _strict_object(manifest_bytes, label)
    artifact, prompt_hashes = _parse_manifest(
        manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        expected_variant=variant,
        expected_fingerprint=entry["decision_prompt_fingerprint"],
    )
    payload = decision_prompt_fingerprint_payload(
        method_variant=variant,
        manifest_sha256=manifest_sha256,
        prompt_file_hashes=prompt_hashes,
    )
    if sha256_json(payload) != entry["decision_prompt_fingerprint"]:
        raise ProductionPromptArtifactError(
            f"{label} decision prompt fingerprint differs from artifact bytes"
        )
    return artifact


def _parse_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    expected_variant: MethodVariant,
    expected_fingerprint: str,
) -> tuple[ProductionDecisionPromptArtifact, dict[str, str]]:
    label = f"prompt manifest {expected_variant.value}"
    _require_exact_fields(manifest, _MANIFEST_FIELDS, label)
    if manifest["kind"] != PROMPT_ARTIFACT_KIND:
        raise ProductionPromptArtifactError(f"{label} kind is invalid")
    if manifest["schema_version"] != PROMPT_ARTIFACT_SCHEMA_VERSION:
        raise ProductionPromptArtifactError(
            f"{label} schema_version is unsupported"
        )
    if manifest["method_variant"] != expected_variant.value:
        raise ProductionPromptArtifactError(
            f"{label} method_variant differs from registry"
        )

    gates_value = manifest["memory_gates"]
    if not isinstance(gates_value, Mapping):
        raise ProductionPromptArtifactError(f"{label}.memory_gates must be an object")
    gates_mapping = dict(gates_value)
    _require_exact_fields(gates_mapping, _MEMORY_GATE_FIELDS, f"{label}.memory_gates")
    if not all(type(value) is bool for value in gates_mapping.values()):
        raise ProductionPromptArtifactError(
            f"{label}.memory_gates values must be booleans"
        )
    memory_gates = PromptMemoryGates(**gates_mapping)

    common_runtime_roles = _string_tuple(
        manifest["common_runtime_roles"],
        f"{label}.common_runtime_roles",
    )
    forbidden_roles = _string_tuple(
        manifest["forbidden_method_roles"],
        f"{label}.forbidden_method_roles",
    )
    roles_value = manifest["roles"]
    if not isinstance(roles_value, list):
        raise ProductionPromptArtifactError(f"{label}.roles must be an array")
    roles: list[ProductionPromptRole] = []
    prompt_hashes: dict[str, str] = {}
    for index, value in enumerate(roles_value):
        role_label = f"{label}.roles[{index}]"
        if not isinstance(value, Mapping):
            raise ProductionPromptArtifactError(f"{role_label} must be an object")
        role_value = dict(value)
        _require_exact_fields(role_value, _ROLE_FIELDS, role_label)
        for field in ("role", "prompt_base", "output_contract"):
            if not isinstance(role_value[field], str):
                raise ProductionPromptArtifactError(
                    f"{role_label}.{field} must be a string"
                )
        prompt_relpath = role_value["prompt_file"]
        if prompt_relpath is None:
            prompt_text = ""
            prompt_sha256 = None
        else:
            if not isinstance(prompt_relpath, str):
                raise ProductionPromptArtifactError(
                    f"{role_label}.prompt_file must be a string or null"
                )
            prompt_path = _safe_regular_file(
                manifest_path.parent,
                prompt_relpath,
                role_label,
            )
            prompt_bytes = _read_bytes(prompt_path, role_label)
            prompt_text = _decode_prompt(prompt_bytes, role_label)
            prompt_sha256 = _sha256_bytes(prompt_bytes)
            previous = prompt_hashes.setdefault(prompt_relpath, prompt_sha256)
            if previous != prompt_sha256:
                raise ProductionPromptArtifactError(
                    f"{role_label} reuses a prompt path with different bytes"
                )
        roles.append(
            ProductionPromptRole(
                role=role_value["role"],
                prompt_base=role_value["prompt_base"],
                output_contract=role_value["output_contract"],
                prompt_relpath=prompt_relpath,
                prompt_text=prompt_text,
                prompt_sha256=prompt_sha256,
            )
        )

    for field in (
        "artifact_scope",
        "policy_representation",
        "shared_prompt_contract_binding",
        "frozen_v83_prompt_binding",
        "decision_structure",
    ):
        if not isinstance(manifest[field], str):
            raise ProductionPromptArtifactError(f"{label}.{field} must be a string")
    for field in ("hierarchical_evolution", "durable_dynamic_memory"):
        if type(manifest[field]) is not bool:
            raise ProductionPromptArtifactError(f"{label}.{field} must be a boolean")

    artifact = ProductionDecisionPromptArtifact(
        method_variant=expected_variant,
        artifact_scope=manifest["artifact_scope"],
        policy_representation=manifest["policy_representation"],
        shared_prompt_contract_binding=manifest[
            "shared_prompt_contract_binding"
        ],
        frozen_v83_prompt_binding=manifest["frozen_v83_prompt_binding"],
        decision_structure=manifest["decision_structure"],
        hierarchical_evolution=manifest["hierarchical_evolution"],
        durable_dynamic_memory=manifest["durable_dynamic_memory"],
        memory_gates=memory_gates,
        common_runtime_roles=common_runtime_roles,
        forbidden_method_roles=forbidden_roles,
        roles=tuple(roles),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        decision_prompt_fingerprint=expected_fingerprint,
    )
    _validate_expected_artifact(artifact)
    return artifact, prompt_hashes


def _validate_expected_artifact(
    artifact: ProductionDecisionPromptArtifact,
) -> None:
    expected = _EXPECTED[artifact.method_variant]
    actual_roles = tuple(
        (role.role, role.prompt_base, role.output_contract)
        for role in artifact.roles
    )
    checks = {
        "artifact_scope": (
            artifact.artifact_scope,
            PRODUCTION_ARTIFACT_SCOPE,
        ),
        "policy_representation": (
            artifact.policy_representation,
            PRODUCTION_POLICY_REPRESENTATION,
        ),
        "shared_prompt_contract_binding": (
            artifact.shared_prompt_contract_binding,
            SHARED_PROMPT_CONTRACT_BINDING,
        ),
        "frozen_v83_prompt_binding": (
            artifact.frozen_v83_prompt_binding,
            FROZEN_V83_PROMPT_BINDING,
        ),
        "decision_structure": (
            artifact.decision_structure,
            expected.decision_structure,
        ),
        "hierarchical_evolution": (
            artifact.hierarchical_evolution,
            expected.hierarchical_evolution,
        ),
        "durable_dynamic_memory": (
            artifact.durable_dynamic_memory,
            expected.durable_dynamic_memory,
        ),
        "memory_gates": (artifact.memory_gates, expected.memory_gates),
        "common_runtime_roles": (
            artifact.common_runtime_roles,
            _COMMON_RUNTIME_ROLES,
        ),
        "forbidden_method_roles": (
            artifact.forbidden_method_roles,
            expected.forbidden_roles,
        ),
        "roles": (actual_roles, expected.roles),
    }
    mismatches = tuple(name for name, pair in checks.items() if pair[0] != pair[1])
    if mismatches:
        raise ProductionPromptArtifactError(
            f"{artifact.method_variant.value} prompt contract differs in: "
            + ", ".join(mismatches)
        )
    if artifact.method_variant is MethodVariant.NO_DYNAMIC_EXPERIENCE:
        if any(role.prompt_relpath is not None for role in artifact.roles):
            raise ProductionPromptArtifactError(
                "No Dynamic Experience must reuse V83 role prompts byte-for-byte"
            )
    elif any(role.prompt_relpath is None for role in artifact.roles if role.prompt_base == "none"):
        raise ProductionPromptArtifactError(
            f"{artifact.method_variant.value} has a standalone role without prompt bytes"
        )


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProductionPromptArtifactError(f"cannot decode {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProductionPromptArtifactError(f"{label} root must be an object")
    return dict(value)


def _decode_prompt(raw: bytes, label: str) -> str:
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProductionPromptArtifactError(f"{label} is not UTF-8") from exc
    if not value.endswith("\n") or "\r" in value or "\x00" in value:
        raise ProductionPromptArtifactError(
            f"{label} must be LF-terminated UTF-8 text without CR/NUL"
        )
    lowered = value.lower()
    forbidden = ("https://", "http://", "authorization: bearer")
    secret_like = re.search(r"(?<![a-z0-9])sk-[a-z0-9]{8,}", lowered)
    if any(token in lowered for token in forbidden) or secret_like is not None:
        raise ProductionPromptArtifactError(
            f"{label} contains provider transport or credential material"
        )
    return value


def _string_tuple(value: object, label: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ProductionPromptArtifactError(
            f"{label} must be an array of non-empty strings"
        )
    return tuple(value)


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    missing = sorted(expected.difference(value))
    unknown = sorted(set(value).difference(expected))
    if missing or unknown:
        raise ProductionPromptArtifactError(
            f"{label} fields differ: missing={missing}, unknown={unknown}"
        )


def _require_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        path = path.resolve()
    if path.is_symlink() or not path.is_dir():
        raise ProductionPromptArtifactError(
            f"{label} must be a regular, non-symlink directory"
        )
    return path.resolve()


def _safe_regular_file(root: Path, relpath: str, label: str) -> Path:
    _require_safe_relpath(relpath, label)
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(relpath).parts)
    cursor = root
    for part in PurePosixPath(relpath).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ProductionPromptArtifactError(
                f"{label} must not traverse a symlink"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProductionPromptArtifactError(f"cannot resolve {label}: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise ProductionPromptArtifactError(f"{label} escapes its artifact root")
    if not candidate.is_file():
        raise ProductionPromptArtifactError(
            f"{label} must be a regular, non-symlink file"
        )
    return candidate


def _require_safe_relpath(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ProductionPromptArtifactError(f"{label} path must be non-empty")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or any(part in ("", ".") for part in path.parts)
    ):
        raise ProductionPromptArtifactError(
            f"{label} path must be a normalized relative POSIX path"
        )


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ProductionPromptArtifactError(f"cannot read {label}: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


__all__ = [
    "build_production_prompt_identity",
    "decision_prompt_fingerprint_payload",
    "load_production_prompt_registry",
]
