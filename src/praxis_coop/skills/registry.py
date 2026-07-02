"""REMAC-inspired command-to-skill registry.

The registry is the boundary that prevents the orchestrator/coder from emitting
arbitrary robot code. Every phase must map to an allowed skill with explicit
preconditions, postconditions, and runtime guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    skill_type: str
    command_schema: dict[str, str]
    allowed_roles: list[str]
    preconditions: list[str]
    postconditions: list[str]
    guards: list[str]
    failure_modes: list[str] = field(default_factory=list)


@dataclass
class SkillRegistry:
    skills: dict[str, SkillDefinition]

    def get(self, name: str) -> SkillDefinition:
        if name not in self.skills:
            raise KeyError(f"Unregistered skill: {name}")
        return self.skills[name]

    def names(self) -> list[str]:
        return sorted(self.skills)


def default_g1_skill_registry() -> SkillRegistry:
    return SkillRegistry(
        skills={
            "walk_to_pose": SkillDefinition(
                name="walk_to_pose",
                skill_type="sonic_locomotion",
                command_schema={"target_pose": "Pose2D", "max_speed_mps": "float"},
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["robot_is_balanced", "local_path_is_clear"],
                postconditions=["robot_at_target_pose"],
                guards=["stop_if_obstacle_too_close", "stop_if_robot_fall"],
                failure_modes=["collision", "robot_fall", "timeout"],
            ),
            "move_lateral": SkillDefinition(
                name="move_lateral",
                skill_type="sonic_ground_plane_locomotion",
                command_schema={
                    "direction": "'left'|'right'|signed_float",
                    "distance_m": "float",
                    "max_speed_mps": "float",
                    "reference_frame": "'body'|'world'",
                },
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["robot_is_balanced", "local_path_is_clear"],
                postconditions=["robot_lateral_offset_reached"],
                guards=["stop_if_obstacle_too_close", "stop_if_robot_fall", "stop_if_partner_lag_large"],
                failure_modes=["collision", "robot_fall", "timeout", "partner_lag"],
            ),
            "move_longitudinal": SkillDefinition(
                name="move_longitudinal",
                skill_type="sonic_ground_plane_locomotion",
                command_schema={
                    "direction": "'forward'|'backward'|signed_float",
                    "distance_m": "float",
                    "max_speed_mps": "float",
                    "reference_frame": "'body'|'world'",
                },
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["robot_is_balanced", "local_path_is_clear"],
                postconditions=["robot_longitudinal_offset_reached"],
                guards=["stop_if_obstacle_too_close", "stop_if_robot_fall", "stop_if_partner_lag_large"],
                failure_modes=["collision", "robot_fall", "timeout", "partner_lag"],
            ),
            "turn_body_yaw_360": SkillDefinition(
                name="turn_body_yaw_360",
                skill_type="sonic_yaw_locomotion",
                command_schema={
                    "delta_yaw_rad": "float in [-2*pi, 2*pi]",
                    "target_yaw_rad": "float optional",
                    "max_yaw_rate_radps": "float",
                    "segment_yaw_rad": "float",
                },
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["robot_is_balanced", "yaw_path_is_clear"],
                postconditions=["robot_yaw_reached"],
                guards=["stop_if_robot_fall", "stop_if_contact_slips", "stop_if_partner_lag_large"],
                failure_modes=["robot_fall", "timeout", "yaw_overshoot", "contact_loss"],
            ),
            "establish_payload_contact": SkillDefinition(
                name="establish_payload_contact",
                skill_type="payload_contact",
                command_schema={"contact_mode": "str", "force_threshold": "float"},
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["near_payload_contact_pose"],
                postconditions=["both_agents_in_contact"],
                guards=["abort_if_force_too_high", "stop_if_contact_slips"],
                failure_modes=["contact_loss", "adapter_error"],
            ),
            "hold_initial_payload_hand_pose": SkillDefinition(
                name="hold_initial_payload_hand_pose",
                skill_type="sonic_upper_body_pose_lock",
                command_schema={
                    "reference_source": "str",
                    "lock_frame": "str",
                    "wrist_pose_tolerance_m": "float",
                    "hand_joint_policy": "str",
                },
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["both_agents_in_contact", "initial_scene_pose_available"],
                postconditions=["initial_hand_pose_locked"],
                guards=[
                    "stop_if_wrist_drift_large",
                    "stop_if_contact_slips",
                    "stop_if_robot_fall",
                ],
                failure_modes=["wrist_drift", "contact_loss", "robot_fall", "adapter_error"],
            ),
            "align_payload_to_corridor": SkillDefinition(
                name="align_payload_to_corridor",
                skill_type="cooperative_payload_alignment",
                command_schema={"yaw_tolerance_deg": "float", "max_speed_mps": "float"},
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["both_agents_in_contact"],
                postconditions=["payload_yaw_aligned"],
                guards=["stop_if_clearance_low", "stop_if_partner_lag_large"],
                failure_modes=["yaw_misalignment", "clearance_violation", "contact_loss"],
            ),
            "carry_payload_path": SkillDefinition(
                name="carry_payload_path",
                skill_type="cooperative_transport",
                command_schema={
                    "object_path": "list[Pose2D]",
                    "robot_paths": "dict[str, list[Pose2D]]",
                    "max_speed_mps": "float",
                },
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["payload_yaw_aligned", "both_agents_in_contact"],
                postconditions=["payload_at_target"],
                guards=[
                    "stop_if_collision",
                    "stop_if_contact_loss",
                    "stop_if_partner_lag_large",
                    "stop_if_progress_stagnates",
                ],
                failure_modes=["collision", "contact_loss", "stagnation", "target_miss"],
            ),
            "stop_and_resync": SkillDefinition(
                name="stop_and_resync",
                skill_type="cooperative_recovery",
                command_schema={"reason": "str"},
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["runtime_guard_violation"],
                postconditions=["both_agents_ready"],
                guards=["hold_payload_stable"],
                failure_modes=["timeout", "contact_loss"],
            ),
            "release_payload": SkillDefinition(
                name="release_payload",
                skill_type="payload_release",
                command_schema={},
                allowed_roles=["navigation_leader", "stability_follower"],
                preconditions=["payload_at_target"],
                postconditions=["payload_stable_and_robots_clear"],
                guards=["stop_if_payload_tilt_large"],
                failure_modes=["target_miss", "collision"],
            ),
        }
    )
