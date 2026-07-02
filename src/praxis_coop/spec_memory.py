"""Executable specification memory."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import ExecutableSpec, TaskTaxonomy


def default_specs() -> list[ExecutableSpec]:
    return [
        ExecutableSpec(
            spec_id="narrow_gate_pre_alignment_v1",
            task_context={
                "physical_coupling": "rigid_object",
                "environment_constraint": "narrow_gate",
            },
            rule_text=(
                "Before entering a narrow passage, align object yaw with the "
                "gate axis and slow down until clearance is verified."
            ),
            trigger=["distance(object_com, gate_entry) < 1.0m"],
            executable_predicate=[
                "abs(object_yaw - gate_axis_yaw) < 7deg",
                "min_clearance(object, gate_edges) > 0.12m",
                "object_speed < 0.15m/s",
            ],
            fallback=[
                "stop_and_realign",
                "reverse_to_pre_gate_pose",
                "switch_pivot_anchor",
            ],
            confidence=0.65,
        ),
        ExecutableSpec(
            spec_id="front_back_contact_sync_v1",
            task_context={
                "physical_coupling": "rigid_object",
                "role_symmetry": "leader_follower",
            },
            rule_text=(
                "The leader may advance only when both robots report stable "
                "object contact and relative distance error is bounded."
            ),
            trigger=["phase in [transport_to_gate, narrow_passage]"],
            executable_predicate=[
                "both_agents_in_contact",
                "front_back_distance_error < 0.08m",
            ],
            fallback=["synchronized_stop", "reestablish_contact"],
            confidence=0.6,
        ),
        ExecutableSpec(
            spec_id="synchronized_lateral_transport_v1",
            task_context={
                "physical_coupling": "rigid_object",
                "object_type": "long_box",
                "interaction_type": "pure_cooperation",
            },
            rule_text=(
                "When two G1 robots carry a long box through lateral motion or "
                "turning, first stabilize dual contact, then send synchronized "
                "lateral commands; do not let one robot advance and the other "
                "compensate afterward."
            ),
            trigger=[
                "transport phase includes lateral movement or turn",
                "both robots are assigned to one shared rigid payload",
            ],
            executable_predicate=[
                "both_agents_in_contact",
                "relative_robot_object_offset_error < threshold",
                "lateral_velocity_commands_are_synchronized",
                "object_yaw_error_remains_bounded",
            ],
            fallback=[
                "synchronized_stop",
                "resync_contact",
                "lower_lateral_speed",
                "transport_object_path with synchronized lateral command",
            ],
            confidence=0.6,
            evidence={
                "failure_cases": [],
                "success_after_patch": [],
            },
        ),
        ExecutableSpec(
            spec_id="avoid_repeated_pose_adjustment_without_progress_v1",
            task_context={
                "physical_coupling": "rigid_object",
                "object_type": "long_box",
                "interaction_type": "pure_cooperation",
            },
            rule_text=(
                "If repeated approach or pre-align actions do not reduce target "
                "distance or move the payload, stop adding more pose adjustment; "
                "resynchronize contact once and switch to direct synchronized "
                "transport along the object path."
            ),
            trigger=[
                "repeated approach/pre_align phases",
                "target distance does not decrease",
                "object velocity near zero",
            ],
            executable_predicate=[
                "payload_progress < threshold over recent window",
                "pose_adjustment_count > limit",
            ],
            fallback=[
                "synchronized_stop",
                "resync_contact once",
                "switch_to_direct_corridor_or_lateral_transport",
                "reduce_speed_and_rerun",
            ],
            confidence=0.55,
            evidence={
                "failure_cases": ["repeated_pose_adjustment_without_payload_progress"],
                "success_after_patch": [],
            },
        ),
    ]


@dataclass
class SpecMemory:
    specs: list[ExecutableSpec] = field(default_factory=default_specs)

    def retrieve(self, taxonomy: TaskTaxonomy, limit: int = 3) -> list[ExecutableSpec]:
        scored: list[tuple[int, ExecutableSpec]] = []
        task = taxonomy.__dict__
        for spec in self.specs:
            score = sum(1 for key, value in spec.task_context.items() if task.get(key) == value)
            if score:
                scored.append((score, spec))
        scored.sort(key=lambda item: (item[0], item[1].confidence), reverse=True)
        return [spec for _, spec in scored[:limit]]

    def promote(self, spec: ExecutableSpec) -> None:
        existing = {item.spec_id: idx for idx, item in enumerate(self.specs)}
        if spec.spec_id in existing:
            self.specs[existing[spec.spec_id]] = spec
        else:
            self.specs.append(spec)
