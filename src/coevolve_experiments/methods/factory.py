"""Single adapter factory used by the common experiment runner."""

from __future__ import annotations

from coevolve_experiments.contracts import MethodVariant

from .base import AdapterMemoryPort, MethodAdapter, RoleBackend
from .coherent import CoherentAdaptedAdapter
from .flat import FlatReflectiveRepairAdapter
from .hierarchical import FullCoEvolveAdapter, NoDynamicExperienceAdapter
from .remac import RemacAdaptedAdapter


FROZEN_REMAC_POSTCONDITION_RETRIES = 2
FROZEN_COHERENT_FEASIBILITY_ADJUSTMENTS = 2


def create_method_adapter(
    variant: MethodVariant | str,
    backend: RoleBackend,
    *,
    memory: AdapterMemoryPort | None = None,
    **unregistered_options: object,
) -> MethodAdapter:
    if unregistered_options:
        names = ", ".join(sorted(unregistered_options))
        raise ValueError(
            "adapter options are frozen by the experiment protocol; "
            f"unregistered override(s): {names}"
        )
    selected = variant if isinstance(variant, MethodVariant) else MethodVariant(variant)
    if selected is MethodVariant.COEVOLE_FULL:
        return FullCoEvolveAdapter(backend, memory=memory)
    if selected is MethodVariant.NO_DYNAMIC_EXPERIENCE:
        return NoDynamicExperienceAdapter(backend, memory=memory)
    if selected is MethodVariant.FLAT_REFLECTIVE_REPAIR:
        return FlatReflectiveRepairAdapter(backend, memory=memory)
    if selected is MethodVariant.REMAC_ADAPTED:
        return RemacAdaptedAdapter(
            backend,
            memory=memory,
            max_postcondition_retries=FROZEN_REMAC_POSTCONDITION_RETRIES,
        )
    if selected is MethodVariant.COHERENT_ADAPTED:
        return CoherentAdaptedAdapter(
            backend,
            memory=memory,
            max_feasibility_adjustments=FROZEN_COHERENT_FEASIBILITY_ADJUSTMENTS,
        )
    raise ValueError(f"unsupported method variant: {selected}")
