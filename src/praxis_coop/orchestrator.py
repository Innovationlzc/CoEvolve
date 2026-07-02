"""High-level orchestration entry point."""

from __future__ import annotations

from dataclasses import dataclass

from .models import SceneGraph, SkillGraph, TransportPlan
from .planner import plan_transport_paths
from .roles import allocate_roles
from .skill_graph import generate_skill_graph
from .spec_memory import SpecMemory
from .taxonomy import classify_task


@dataclass
class OrchestrationResult:
    graph: SkillGraph
    object_path_len: int
    robot_path_lens: dict[str, int]
    retrieved_specs: list[str]
    transport_plan: TransportPlan


def build_initial_skill_graph(
    instruction: str,
    scene: SceneGraph,
    memory: SpecMemory | None = None,
) -> OrchestrationResult:
    memory = memory or SpecMemory()
    taxonomy = classify_task(instruction, scene)
    specs = memory.retrieve(taxonomy)
    roles = allocate_roles(scene, taxonomy)
    transport_plan = plan_transport_paths(scene, roles)
    graph = generate_skill_graph(taxonomy, roles, specs, transport_plan)
    return OrchestrationResult(
        graph=graph,
        object_path_len=len(transport_plan.object_path),
        robot_path_lens={
            robot_id: len(path) for robot_id, path in transport_plan.robot_paths.items()
        },
        retrieved_specs=[spec.spec_id for spec in specs],
        transport_plan=transport_plan,
    )
