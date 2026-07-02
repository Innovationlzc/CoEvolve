"""Experiment helpers for Praxis-Coop SkillGraph patch application."""

from __future__ import annotations

from copy import deepcopy

from .models import ContactFlowSpec, ContactIntent, PatchProposal, SkillGraph, SkillNode


def apply_patch(graph: SkillGraph, patch: PatchProposal) -> SkillGraph:
    """Apply a small, typed patch to a SkillGraph copy."""

    patched = deepcopy(graph)
    if patch.patch_type == "parameter":
        node_name = patch.target.split(".", 1)[0]
        node = _find_node(patched, node_name)
        node.parameters.update(patch.change)
        return patched

    if patch.patch_id == "patch_insert_resync_contact" or (
        patch.patch_type == "graph" and "insert_before" in patch.change and "node" in patch.change
    ):
        target_node = str(patch.change["insert_before"])
        insert_at = _find_node_index(patched, target_node)
        node_name = str(patch.change.get("node", "resync_contact"))
        if not any(node.name == node_name for node in patched.nodes):
            patched.nodes.insert(
                insert_at,
                SkillNode(
                    name=node_name,
                    agents=patched.nodes[insert_at].agents,
                    parameters={"contact_mode": "recenter_and_confirm"},
                    preconditions=["robots_near_object"],
                    postconditions=["both_agents_in_contact"],
                    monitors=["contact_loss"],
                    fallback=["stop_and_reapproach"],
                    timeout_s=10,
                    phase_id=node_name,
                    sync_group=patched.nodes[insert_at].sync_group or "payload_team",
                    contact_flow=ContactFlowSpec(
                        phase_id=node_name,
                        phase_index=max(insert_at, 0),
                        contact=ContactIntent(
                            object_id="payload",
                            mode="recenter_and_confirm",
                            target_region="assigned_payload_face",
                            maintain=True,
                        ),
                        sync_group=patched.nodes[insert_at].sync_group or "payload_team",
                        abort_on_contact_loss=True,
                        verifier_tags=("contact_loss",),
                    ),
                ),
            )
        return patched

    if patch.change.get("action") == "replan_with_lower_speed":
        for node in patched.nodes:
            if "max_speed_mps" in node.parameters:
                node.parameters["max_speed_mps"] = min(float(node.parameters["max_speed_mps"]), 0.10)
        return patched

    return patched


def _find_node(graph: SkillGraph, name: str) -> SkillNode:
    return graph.nodes[_find_node_index(graph, name)]


def _find_node_index(graph: SkillGraph, name: str) -> int:
    for idx, node in enumerate(graph.nodes):
        if node.name == name:
            return idx
    raise ValueError(f"Skill node not found: {name}")
