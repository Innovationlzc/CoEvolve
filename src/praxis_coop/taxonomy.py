"""Task taxonomy classification."""

from __future__ import annotations

from .models import SceneGraph, TaskTaxonomy


def classify_task(instruction: str, scene: SceneGraph) -> TaskTaxonomy:
    text = instruction.lower()
    has_gate = scene.gate is not None
    object_length = scene.movable_object.size[0]
    object_width = scene.movable_object.size[1]
    object_type = "long_box" if object_length > 1.5 * object_width else "box"

    if "scout" in text or "camera" in text or "observe" in text:
        observability = "asymmetric_vision"
    elif "occlusion" in text or "blocked" in text:
        observability = "occlusion"
    else:
        observability = "full_state"

    if "insert" in text or "shelf" in text or "target" in text:
        objective = "transport_insert"
    else:
        objective = "transport"

    return TaskTaxonomy(
        physical_coupling="rigid_object",
        role_symmetry="leader_follower",
        object_type=object_type,
        environment_constraint="narrow_gate" if has_gate else "open",
        observability=observability,
        objective=objective,
    )
