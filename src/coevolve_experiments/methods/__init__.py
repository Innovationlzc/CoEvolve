"""Isolated method adapters for Table-2 comparison and ablations."""

from coevolve_experiments.live import ExecutionCapability, PolicyRepresentation

from .backend import RoleCall, ScriptedRoleBackend
from .base import (
    ActionExemplarQuery,
    AdapterContext,
    AdapterContractError,
    AdapterMemoryPort,
    AdapterMemoryLifecycle,
    AdapterUsageSummary,
    MemorySwitches,
    MethodAdapter,
    MethodDescriptor,
    MethodRole,
    NullAdapterMemory,
    OrchestrationResult,
    REQUIRED_SAFETY_INVARIANTS,
    RoleBackend,
    RoleResponse,
    RoleUsage,
    SafetyInvariants,
    flat_strategy_commit_from_response,
    orchestration_result_from_response,
)
from .coherent import CoherentAdaptedAdapter
from .factory import (
    FROZEN_COHERENT_FEASIBILITY_ADJUSTMENTS,
    FROZEN_REMAC_POSTCONDITION_RETRIES,
    create_method_adapter,
)
from .flat import FlatReflectiveRepairAdapter
from .hierarchical import FullCoEvolveAdapter, NoDynamicExperienceAdapter
from .provenance import (
    COHERENT_PROVENANCE,
    COHERENT_REPOSITORY_COMMIT,
    REMAC_PROVENANCE,
    REMAC_REPOSITORY_COMMIT,
    AdaptationProvenance,
)
from .remac import RemacAdaptedAdapter

__all__ = [
    "ActionExemplarQuery",
    "AdapterContext",
    "AdapterContractError",
    "AdapterMemoryPort",
    "AdapterMemoryLifecycle",
    "AdapterUsageSummary",
    "AdaptationProvenance",
    "COHERENT_PROVENANCE",
    "COHERENT_REPOSITORY_COMMIT",
    "CoherentAdaptedAdapter",
    "ExecutionCapability",
    "FlatReflectiveRepairAdapter",
    "FROZEN_COHERENT_FEASIBILITY_ADJUSTMENTS",
    "FROZEN_REMAC_POSTCONDITION_RETRIES",
    "FullCoEvolveAdapter",
    "MemorySwitches",
    "MethodAdapter",
    "MethodDescriptor",
    "MethodRole",
    "NoDynamicExperienceAdapter",
    "NullAdapterMemory",
    "OrchestrationResult",
    "REMAC_PROVENANCE",
    "REMAC_REPOSITORY_COMMIT",
    "PolicyRepresentation",
    "REQUIRED_SAFETY_INVARIANTS",
    "RemacAdaptedAdapter",
    "RoleBackend",
    "RoleCall",
    "RoleResponse",
    "RoleUsage",
    "SafetyInvariants",
    "ScriptedRoleBackend",
    "flat_strategy_commit_from_response",
    "orchestration_result_from_response",
    "create_method_adapter",
]
