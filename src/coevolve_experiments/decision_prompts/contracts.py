"""Immutable contracts for external production decision-prompt artifacts.

The adapters in :mod:`coevolve_experiments.methods` deliberately emit the
small ``TeamIntent`` smoke IR.  The records in this module describe a separate
artifact family for a future production adapter that continues to emit the
frozen Praxis ``PolicyScript``/``RobotPolicyScript`` representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping, Tuple

from ..contracts import MethodDecisionPromptPin, MethodVariant


PROMPT_ARTIFACT_KIND = "coevolve_production_decision_prompt_artifact"
PROMPT_ARTIFACT_SCHEMA_VERSION = 1
PROMPT_REGISTRY_KIND = "coevolve_production_decision_prompt_registry"
PROMPT_REGISTRY_SCHEMA_VERSION = 1
PROMPT_FINGERPRINT_KIND = "coevolve_production_decision_prompt_fingerprint"
PROMPT_FINGERPRINT_SCHEMA_VERSION = 1

PRODUCTION_ARTIFACT_SCOPE = "production_live_decision_roles"
PRODUCTION_POLICY_REPRESENTATION = "praxis_code_as_policy"
SHARED_PROMPT_CONTRACT_BINDING = "lock://shared_prompt_contract_fingerprint"
FROZEN_V83_PROMPT_BINDING = (
    "lock://full_method_compatibility.prompt_sha256"
)
FROZEN_V83_FULL_DECISION_PROMPT_SHA256 = (
    "7214a561d9338956f59d5f658d6a4d3e3716d5dc594e475cbbce7998624680ec"
)
FROZEN_V83_PROMPT_BASE_SYMBOLS: Mapping[str, str] = MappingProxyType(
    {
        "frozen_v83_task_orchestrator": "TASK_ORCHESTRATOR_SYSTEM_PROMPT",
        "frozen_v83_policy_coder": "POLICY_CODER_SYSTEM_PROMPT",
        "frozen_v83_vlm_failure_analyzer": (
            "VLM_FAILURE_ANALYZER_SYSTEM_PROMPT"
        ),
        "frozen_v83_evolution_manager": "EVOLUTION_MANAGER_SYSTEM_PROMPT",
        "frozen_v83_mutation_operator": "MUTATION_OPERATOR_SYSTEM_PROMPT",
        "frozen_v83_code_repair": "CODE_REPAIR_SYSTEM_PROMPT",
        "frozen_v83_memory_update": "MEMORY_UPDATE_SYSTEM_PROMPT",
    }
)

EXTERNAL_PROMPT_VARIANTS: Tuple[MethodVariant, ...] = (
    MethodVariant.NO_DYNAMIC_EXPERIENCE,
    MethodVariant.FLAT_REFLECTIVE_REPAIR,
    MethodVariant.REMAC_ADAPTED,
    MethodVariant.COHERENT_ADAPTED,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProductionPromptArtifactError(ValueError):
    """A production prompt artifact is missing, ambiguous, or inconsistent."""


@dataclass(frozen=True, slots=True)
class PromptMemoryGates:
    """The only dynamic-experience switches represented by an artifact."""

    strategy_retrieval: bool
    strategy_promotion: bool
    action_exemplar_retrieval: bool
    action_exemplar_promotion: bool
    memory_writer_call: bool

    def __post_init__(self) -> None:
        if not all(
            type(value) is bool
            for value in (
                self.strategy_retrieval,
                self.strategy_promotion,
                self.action_exemplar_retrieval,
                self.action_exemplar_promotion,
                self.memory_writer_call,
            )
        ):
            raise ProductionPromptArtifactError(
                "production prompt memory gates must be booleans"
            )

    @property
    def all_disabled(self) -> bool:
        return not any(
            (
                self.strategy_retrieval,
                self.strategy_promotion,
                self.action_exemplar_retrieval,
                self.action_exemplar_promotion,
                self.memory_writer_call,
            )
        )


@dataclass(frozen=True, slots=True)
class ProductionPromptRole:
    """One model role and its optional overlay on a hash-bound V83 prompt."""

    role: str
    prompt_base: str
    output_contract: str
    prompt_relpath: str | None
    prompt_text: str
    prompt_sha256: str | None

    def __post_init__(self) -> None:
        for name in ("role", "prompt_base", "output_contract"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProductionPromptArtifactError(
                    f"production prompt role {name} must be non-empty"
                )
        if self.prompt_relpath is None:
            if self.prompt_text or self.prompt_sha256 is not None:
                raise ProductionPromptArtifactError(
                    "a role without an overlay file cannot contain overlay bytes"
                )
            if self.prompt_base == "none":
                raise ProductionPromptArtifactError(
                    "a standalone role requires a prompt file"
                )
        else:
            if not self.prompt_text.strip():
                raise ProductionPromptArtifactError(
                    f"prompt overlay is empty for role {self.role}"
                )
            _require_sha256(
                f"{self.role} prompt_sha256",
                self.prompt_sha256,
            )

    @property
    def uses_frozen_v83_base(self) -> bool:
        return self.prompt_base != "none"


@dataclass(frozen=True, slots=True)
class ProductionDecisionPromptArtifact:
    """A fully validated, byte-bound method-specific prompt bundle."""

    method_variant: MethodVariant
    artifact_scope: str
    policy_representation: str
    shared_prompt_contract_binding: str
    frozen_v83_prompt_binding: str
    decision_structure: str
    hierarchical_evolution: bool
    durable_dynamic_memory: bool
    memory_gates: PromptMemoryGates
    common_runtime_roles: Tuple[str, ...]
    forbidden_method_roles: Tuple[str, ...]
    roles: Tuple[ProductionPromptRole, ...]
    manifest_path: Path
    manifest_sha256: str
    decision_prompt_fingerprint: str

    def __post_init__(self) -> None:
        if self.method_variant not in EXTERNAL_PROMPT_VARIANTS:
            raise ProductionPromptArtifactError(
                "external prompt artifacts cannot replace frozen Full CoEvolve"
            )
        if self.artifact_scope != PRODUCTION_ARTIFACT_SCOPE:
            raise ProductionPromptArtifactError(
                "prompt artifact is not scoped to production live decision roles"
            )
        if self.policy_representation != PRODUCTION_POLICY_REPRESENTATION:
            raise ProductionPromptArtifactError(
                "production prompt artifact must use praxis_code_as_policy"
            )
        if (
            self.shared_prompt_contract_binding
            != SHARED_PROMPT_CONTRACT_BINDING
        ):
            raise ProductionPromptArtifactError(
                "shared prompt contract binding differs from the frozen URI"
            )
        if self.frozen_v83_prompt_binding != FROZEN_V83_PROMPT_BINDING:
            raise ProductionPromptArtifactError(
                "V83 prompt binding differs from the frozen compatibility URI"
            )
        if not self.decision_structure.strip():
            raise ProductionPromptArtifactError(
                "decision_structure must be non-empty"
            )
        if type(self.hierarchical_evolution) is not bool or type(
            self.durable_dynamic_memory
        ) is not bool:
            raise ProductionPromptArtifactError(
                "hierarchy and durable-memory flags must be booleans"
            )
        if not isinstance(self.memory_gates, PromptMemoryGates):
            raise ProductionPromptArtifactError(
                "memory_gates must be PromptMemoryGates"
            )
        if not isinstance(self.roles, tuple) or not all(
            isinstance(role, ProductionPromptRole) for role in self.roles
        ):
            raise ProductionPromptArtifactError(
                "roles must be a tuple of ProductionPromptRole"
            )
        if not isinstance(self.common_runtime_roles, tuple) or not all(
            isinstance(role, str) and role.strip()
            for role in self.common_runtime_roles
        ):
            raise ProductionPromptArtifactError(
                "common_runtime_roles must be a tuple of strings"
            )
        if not isinstance(self.forbidden_method_roles, tuple) or not all(
            isinstance(role, str) and role.strip()
            for role in self.forbidden_method_roles
        ):
            raise ProductionPromptArtifactError(
                "forbidden_method_roles must be a tuple of strings"
            )
        role_names = tuple(role.role for role in self.roles)
        if not role_names or len(role_names) != len(set(role_names)):
            raise ProductionPromptArtifactError(
                "prompt role names must be non-empty and unique"
            )
        if len(self.common_runtime_roles) != len(set(self.common_runtime_roles)):
            raise ProductionPromptArtifactError(
                "common runtime roles must be unique"
            )
        if len(self.forbidden_method_roles) != len(
            set(self.forbidden_method_roles)
        ):
            raise ProductionPromptArtifactError(
                "forbidden method roles must be unique"
            )
        if set(role_names).intersection(self.forbidden_method_roles):
            raise ProductionPromptArtifactError(
                "a role cannot be both enabled and forbidden"
            )
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_sha256(
            "decision_prompt_fingerprint",
            self.decision_prompt_fingerprint,
        )

    def role(self, name: str) -> ProductionPromptRole:
        selected = tuple(role for role in self.roles if role.role == name)
        if len(selected) != 1:
            raise ProductionPromptArtifactError(
                f"prompt role does not resolve exactly once: {name!r}"
            )
        return selected[0]


@dataclass(frozen=True, slots=True)
class ProductionPromptRegistry:
    """Closed-world registry of the four post-Table-1 prompt bundles."""

    artifacts: Tuple[ProductionDecisionPromptArtifact, ...]
    registry_path: Path
    registry_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, tuple) or not all(
            isinstance(artifact, ProductionDecisionPromptArtifact)
            for artifact in self.artifacts
        ):
            raise ProductionPromptArtifactError(
                "registry artifacts must be a tuple of prompt artifacts"
            )
        variants = tuple(artifact.method_variant for artifact in self.artifacts)
        if variants != EXTERNAL_PROMPT_VARIANTS:
            raise ProductionPromptArtifactError(
                "prompt registry variants/order differ from the frozen contract"
            )
        _require_sha256("registry_sha256", self.registry_sha256)

    def artifact_for(
        self,
        method_variant: MethodVariant | str,
    ) -> ProductionDecisionPromptArtifact:
        try:
            variant = (
                method_variant
                if isinstance(method_variant, MethodVariant)
                else MethodVariant(method_variant)
            )
        except ValueError as exc:
            raise ProductionPromptArtifactError(
                f"unknown method variant: {method_variant!r}"
            ) from exc
        if variant is MethodVariant.COEVOLE_FULL:
            raise ProductionPromptArtifactError(
                "Full CoEvolve must use the frozen V83 prompts.py byte pin"
            )
        selected = tuple(
            artifact
            for artifact in self.artifacts
            if artifact.method_variant is variant
        )
        if len(selected) != 1:
            raise ProductionPromptArtifactError(
                f"prompt artifact does not resolve exactly once: {variant.value}"
            )
        return selected[0]

    def method_decision_prompt_pins(
        self,
        *,
        full_method_prompt_fingerprint: str,
    ) -> Tuple[MethodDecisionPromptPin, ...]:
        """Return all five pins in canonical ``MethodVariant`` order.

        Full CoEvolve is intentionally supplied by the caller because its
        identity is the raw SHA-256 of the frozen V83 ``prompts.py`` bytes.
        The registry is not allowed to substitute or copy that artifact.
        """

        _require_sha256(
            "full_method_prompt_fingerprint",
            full_method_prompt_fingerprint,
        )
        if (
            full_method_prompt_fingerprint
            != FROZEN_V83_FULL_DECISION_PROMPT_SHA256
        ):
            raise ProductionPromptArtifactError(
                "Full CoEvolve prompt fingerprint is not the frozen V83 byte hash"
            )
        values = {
            artifact.method_variant: artifact.decision_prompt_fingerprint
            for artifact in self.artifacts
        }
        values[MethodVariant.COEVOLE_FULL] = full_method_prompt_fingerprint
        return tuple(
            MethodDecisionPromptPin(variant, values[variant])
            for variant in MethodVariant
        )


@dataclass(frozen=True, slots=True)
class ProductionPromptIdentity:
    """Lock-builder input with no inferred or placeholder prompt hashes."""

    shared_prompt_contract_fingerprint: str
    method_decision_prompt_pins: Tuple[MethodDecisionPromptPin, ...]
    registry_fingerprint: str

    def __post_init__(self) -> None:
        _require_sha256(
            "shared_prompt_contract_fingerprint",
            self.shared_prompt_contract_fingerprint,
        )
        _require_sha256("registry_fingerprint", self.registry_fingerprint)
        if not isinstance(self.method_decision_prompt_pins, tuple) or not all(
            isinstance(pin, MethodDecisionPromptPin)
            for pin in self.method_decision_prompt_pins
        ):
            raise ProductionPromptArtifactError(
                "prompt identity pins must be a tuple of MethodDecisionPromptPin"
            )
        if tuple(pin.method_variant for pin in self.method_decision_prompt_pins) != tuple(
            MethodVariant
        ):
            raise ProductionPromptArtifactError(
                "prompt identity requires one pin in canonical order per method"
            )


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProductionPromptArtifactError(
            f"{name} must be a lowercase 64-character SHA-256 digest"
        )


__all__ = [
    "EXTERNAL_PROMPT_VARIANTS",
    "FROZEN_V83_PROMPT_BINDING",
    "FROZEN_V83_PROMPT_BASE_SYMBOLS",
    "FROZEN_V83_FULL_DECISION_PROMPT_SHA256",
    "PROMPT_ARTIFACT_KIND",
    "PROMPT_ARTIFACT_SCHEMA_VERSION",
    "PROMPT_FINGERPRINT_KIND",
    "PROMPT_FINGERPRINT_SCHEMA_VERSION",
    "PROMPT_REGISTRY_KIND",
    "PROMPT_REGISTRY_SCHEMA_VERSION",
    "PRODUCTION_ARTIFACT_SCOPE",
    "PRODUCTION_POLICY_REPRESENTATION",
    "ProductionDecisionPromptArtifact",
    "ProductionPromptArtifactError",
    "ProductionPromptIdentity",
    "ProductionPromptRegistry",
    "ProductionPromptRole",
    "PromptMemoryGates",
    "SHARED_PROMPT_CONTRACT_BINDING",
]
