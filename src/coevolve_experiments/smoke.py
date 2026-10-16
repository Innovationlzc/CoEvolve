"""Dependency-free adapter smoke runs; these never dispatch robot physics."""

from __future__ import annotations

from typing import Any

from .contracts import MethodVariant
from .methods import MethodRole, ScriptedRoleBackend, create_method_adapter
from .runtime import ExperimentEngine, ScriptedRuntime, VerifierOutcome, make_team_proposal


def smoke_method_adapter(variant: MethodVariant | str) -> dict[str, Any]:
    selected = variant if isinstance(variant, MethodVariant) else MethodVariant(variant)
    proposal = make_team_proposal(f"{selected.value}_terminal")
    committed_plan = {
        "updated_plan": {"plan": "fixed-smoke-plan"},
        "committed_strategy": {
            "primitive": "cooperative_move",
            "semantic_objective": f"{selected.value}_terminal",
            "coordination_mode": "synchronized",
        },
    }
    flat_strategy = dict(committed_plan["committed_strategy"])
    scripts = {
        MethodVariant.COEVOLE_FULL: [
            (MethodRole.TASK_ORCHESTRATOR.value, committed_plan),
            (MethodRole.POLICY_GENERATOR.value, proposal),
            (MethodRole.MEMORY_UPDATE.value, {}),
        ],
        MethodVariant.NO_DYNAMIC_EXPERIENCE: [
            (MethodRole.TASK_ORCHESTRATOR.value, committed_plan),
            (MethodRole.POLICY_GENERATOR.value, proposal),
        ],
        MethodVariant.FLAT_REFLECTIVE_REPAIR: [
            (MethodRole.FLAT_TASK_PLANNER.value, committed_plan),
            (MethodRole.FLAT_STRATEGY_SELECTOR.value, flat_strategy),
            (MethodRole.FLAT_POLICY_GENERATOR.value, proposal),
            (MethodRole.MEMORY_UPDATE.value, {}),
        ],
        MethodVariant.REMAC_ADAPTED: [
            (MethodRole.REMAC_PLANNER.value, {"plan": "paper-level-smoke-plan"}),
            (MethodRole.REMAC_PRECONDITION_REFLECTION.value, proposal),
        ],
        MethodVariant.COHERENT_ADAPTED: [
            (MethodRole.COHERENT_CENTRAL_PROPOSER.value, proposal),
            (MethodRole.COHERENT_FRONT_FEASIBILITY.value, {"feasible": True}),
            (MethodRole.COHERENT_BACK_FEASIBILITY.value, {"feasible": True}),
        ],
    }
    backend = ScriptedRoleBackend(scripts[selected])
    adapter = create_method_adapter(selected, backend)
    runtime = ScriptedRuntime(
        [VerifierOutcome(True, terminal=True, report="scripted terminal")]
    )
    result = ExperimentEngine(adapter, runtime.callbacks()).run(
        "scripted adapter contract smoke"
    )
    backend.assert_exhausted()
    return {
        "method_variant": selected.value,
        "policy_representation": adapter.descriptor.policy_representation.value,
        "execution_capability": adapter.descriptor.execution_capability.value,
        "paper_evidence": False,
        "status": result.status.value,
        "success": result.success,
        "physical_transactions": result.accounting.physical_transactions,
        "accepted_transactions": result.accounting.accepted_transactions,
        "model_calls": result.usage.get("model_calls"),
        "roles": list(backend.roles),
        "physics_dispatched": False,
    }


def smoke_all_method_adapters() -> list[dict[str, Any]]:
    return [smoke_method_adapter(variant) for variant in MethodVariant]


__all__ = ["smoke_all_method_adapters", "smoke_method_adapter"]
