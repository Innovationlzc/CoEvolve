"""Task orchestrator for target -> plan -> subtask -> policy script."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import BoxCarryTask, ObservationBundle, PolicyScript, Subtask
from .static_specs import static_skill_library_doc


@dataclass
class TaskOrchestrator:
    """Deterministic first-pass orchestrator for the box-carry task.

    LLM/VLM prompting can replace the deterministic methods later. The output
    contract stays the same: a plan, subtasks, and executable policy code.
    """

    default_transport_skill: str = "carry_lateral_left"
    default_transport_distance_m: float = 1.0
    default_transport_speed_mps: float = 0.50
    task_stage: str = "stage1_straight"

    def generate_initial_policy(
        self,
        task: BoxCarryTask,
        observation: ObservationBundle,
        *,
        attempt: int = 1,
    ) -> PolicyScript:
        specs = list(observation.retrieved_specs)
        visual_note = _visual_note(observation)
        plan = [
            "stabilize both G1 robots with carry-ready arms in the manual layout",
            visual_note,
            "place the light long_box onto the support pose only after robots are stable",
            f"move toward the target with {self.default_transport_skill} in synchronized room-scale segments",
        ]
        subtasks = [
            Subtask(
                name="place_payload",
                description="Make the deferred long_box visible and colliding on the carry-ready arms.",
                skill_name="place_payload_on_arms",
                parameters={"hold_s": 4.0},
                guard="robots_stable_and_carry_pose_locked",
            ),
            Subtask(
                name="straight_transport_1",
                description="Move both robots toward the target with synchronized SONIC locomotion.",
                skill_name=self.default_transport_skill,
                parameters={
                    "distance_m": self.default_transport_distance_m,
                    "speed_mps": self.default_transport_speed_mps,
                },
                guard="payload_supported_by_both_robots",
            ),
        ]
        if self.task_stage == "full_scene":
            plan.append("continue segmented target-directed transport until the target zone is reached")
            for index in range(2, 5):
                subtasks.append(
                    Subtask(
                        name=f"straight_transport_{index}",
                        description="Continue target-directed transport with verifier-checked room-scale moves.",
                        skill_name=self.default_transport_skill,
                        parameters={
                            "distance_m": self.default_transport_distance_m,
                            "speed_mps": self.default_transport_speed_mps,
                        },
                        guard="payload_supported_by_both_robots_and_no_wall_contact",
                    )
                )
            subtasks.extend(
                [
                    Subtask(
                        name="place_payload_crouch",
                        description="Lower both robots in place to set down the payload near the target.",
                        skill_name="carry_place_crouch",
                        parameters={"place_height": 0.55, "ramp_s": 2.0, "hold_s": 10.0},
                        guard="target_region_reached_and_payload_stable",
                    ),
                    Subtask(
                        name="recover_upright",
                        description="Recover both robots to upright after placement.",
                        skill_name="carry_recover_stand",
                        parameters={"height": 0.80, "ramp_s": 3.0, "hold_s": 5.0},
                        guard="payload_stable_after_place",
                    ),
                ]
            )
        else:
            plan.append("stop for verifier inspection before continuing to later subtasks")
        code = render_policy_code(
            script_id=f"box_carry_attempt_{attempt:02d}",
            task=task,
            plan=tuple(plan),
            subtasks=tuple(subtasks),
            specs=specs,
        )
        return PolicyScript(
            script_id=f"box_carry_attempt_{attempt:02d}",
            attempt=attempt,
            code=code,
            plan=tuple(plan),
            subtasks=tuple(subtasks),
            rationale="Initial manual-envelope straight transport from the static SONIC SDK.",
        )

    def skill_library_prompt(self) -> str:
        return static_skill_library_doc()


def render_policy_code(
    *,
    script_id: str,
    task: BoxCarryTask,
    plan: tuple[str, ...],
    subtasks: tuple[Subtask, ...],
    specs: list[str],
) -> str:
    """Render an executable policy(api) script with comments as trace metadata."""

    lines = [
        f'"""Generated policy script: {script_id}.',
        "",
        f"Target: {task.target}",
        "",
        "Plan:",
        *[f"- {item}" for item in plan],
    ]
    if specs:
        lines.extend(["", "Retrieved deployment specs:", *[f"- {item}" for item in specs]])
    lines.extend(['"""', "", "", "def policy(api):"])
    lines.append("    # The policy can only call the injected safe G1SkillAPI.")
    for subtask in subtasks:
        lines.append("")
        lines.append(f"    # subtask: {subtask.name}")
        lines.append(f"    # guard: {subtask.guard or 'none'}")
        args = ", ".join(f"{key}={value!r}" for key, value in subtask.parameters.items())
        lines.append(f"    api.{_api_method_for_skill(subtask.skill_name)}({args})")
    lines.append("")
    lines.append("    api.stop_all()")
    lines.append("")
    return "\n".join(lines)


def _api_method_for_skill(skill_name: str) -> str:
    mapping = {
        "carry_forward": "carry_forward",
        "carry_backward": "carry_backward",
        "carry_lateral_left": "carry_lateral_left",
        "carry_lateral_right": "carry_lateral_right",
        "carry_turn_clockwise_90": "turn_clockwise",
        "carry_turn_counterclockwise_90": "turn_counterclockwise",
        "carry_turn_in_place": "turn_clockwise",
        "carry_place_crouch": "place_crouch",
        "carry_recover_stand": "recover_stand",
        "place_payload_on_arms": "place_payload_on_arms",
    }
    return mapping.get(skill_name, skill_name)


def _visual_note(observation: ObservationBundle) -> str:
    if not observation.visual_observations:
        return "no external camera observation is available; rely on runtime state verifier"
    visual = observation.visual_observations[-1]
    if visual.frame_path:
        return f"use external wide-view camera frame for grounding: {visual.frame_path}"
    return f"external camera status: {visual.summary}"
