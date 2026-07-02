"""CFgen-lite phase annotation for cooperative SONIC skill graphs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from praxis_coop.models import ContactFlowSpec, ContactIntent, Pose2D, SceneGraph, SkillGraph, SkillNode


@dataclass(frozen=True)
class ContactFlowPhase:
    """One verifier-readable cooperative contact-flow phase."""

    node_name: str
    phase_id: str
    phase_index: int
    sync_group: str
    contact: ContactIntent | None = None
    object_goal: Pose2D | None = None
    abort_on_contact_loss: bool = False
    allowed_contact_loss_s: float = 0.0
    verifier_tags: tuple[str, ...] = ()

    def to_spec(self) -> ContactFlowSpec:
        return ContactFlowSpec(
            phase_id=self.phase_id,
            phase_index=self.phase_index,
            contact=self.contact,
            object_goal=self.object_goal,
            sync_group=self.sync_group,
            abort_on_contact_loss=self.abort_on_contact_loss,
            allowed_contact_loss_s=self.allowed_contact_loss_s,
            verifier_tags=self.verifier_tags,
        )


@dataclass(frozen=True)
class CooperativeContactFlow:
    """Annotated graph plus phase records used by the SONIC adapter."""

    graph: SkillGraph
    phases: list[ContactFlowPhase]


class CooperativeCFGen:
    """Generate contact-flow metadata from a cooperative SkillGraph.

    The output intentionally stays at the SONIC adapter layer: phase ids, contact
    modes, sync groups, object goals, and abort policy. It does not synthesize
    wrist, ankle, or joint references.
    """

    def annotate(self, graph: SkillGraph, scene: SceneGraph) -> CooperativeContactFlow:
        annotated = deepcopy(graph)
        phases: list[ContactFlowPhase] = []
        for index, node in enumerate(annotated.nodes):
            phase = self._phase_for_node(node, scene, index)
            node.phase_id = phase.phase_id
            node.sync_group = phase.sync_group
            node.contact_flow = phase.to_spec()
            phases.append(phase)
        return CooperativeContactFlow(graph=annotated, phases=phases)

    def _phase_for_node(self, node: SkillNode, scene: SceneGraph, index: int) -> ContactFlowPhase:
        if node.contact_flow is not None:
            spec = node.contact_flow
            return ContactFlowPhase(
                node_name=node.name,
                phase_id=spec.phase_id or node.phase_id or node.name,
                phase_index=spec.phase_index,
                sync_group=spec.sync_group or node.sync_group or "payload_team",
                contact=spec.contact,
                object_goal=spec.object_goal,
                abort_on_contact_loss=spec.abort_on_contact_loss,
                allowed_contact_loss_s=spec.allowed_contact_loss_s,
                verifier_tags=spec.verifier_tags,
            )

        sync_group = node.sync_group or str(node.parameters.get("sync_group", "payload_team"))
        phase_id = node.phase_id or str(node.parameters.get("phase_id", node.name))
        object_goal = _node_object_goal(node, scene)
        contact = _node_contact_intent(node, scene)
        abort_on_contact_loss = bool("contact_loss" in node.monitors or contact.maintain)
        allowed_contact_loss_s = float(node.parameters.get("allowed_contact_loss_s", 0.0))
        return ContactFlowPhase(
            node_name=node.name,
            phase_id=phase_id,
            phase_index=index,
            sync_group=sync_group,
            contact=contact,
            object_goal=object_goal,
            abort_on_contact_loss=abort_on_contact_loss,
            allowed_contact_loss_s=allowed_contact_loss_s,
            verifier_tags=tuple(node.monitors),
        )


def _node_contact_intent(node: SkillNode, scene: SceneGraph) -> ContactIntent:
    object_id = str(node.parameters.get("object_id", scene.movable_object.object_id))
    if node.name in {"establish_contact", "resync_contact"}:
        mode = str(node.parameters.get("contact_mode", "push_or_grasp"))
        return ContactIntent(
            object_id=object_id,
            mode=mode,
            target_region="assigned_payload_face",
            maintain=True,
        )
    if node.name == "lock_initial_hand_pose":
        return ContactIntent(
            object_id=object_id,
            mode="hold_initial_payload_hand_pose",
            target_region=str(node.parameters.get("lock_frame", "initial_pelvis_relative_pose")),
            maintain=True,
        )
    if node.name in {"pre_align_object", "transport_object_path"}:
        return ContactIntent(
            object_id=object_id,
            mode=str(node.parameters.get("contact_mode", "maintain_payload_contact")),
            target_region=str(node.parameters.get("path_strategy", "planned_object_path")),
            maintain=True,
        )
    if node.name == "release_and_exit":
        return ContactIntent(object_id=object_id, mode="release", links=())
    return ContactIntent(object_id=object_id, mode=str(node.parameters.get("contact_intent", "none")), links=())


def _node_object_goal(node: SkillNode, scene: SceneGraph) -> Pose2D | None:
    if node.name in {"pre_align_object", "transport_object_path", "release_and_exit"}:
        return scene.target.pose
    payload = node.parameters.get("object_goal")
    if isinstance(payload, dict) and "x" in payload and "y" in payload:
        return Pose2D(float(payload["x"]), float(payload["y"]), float(payload.get("yaw", 0.0)))
    return None
