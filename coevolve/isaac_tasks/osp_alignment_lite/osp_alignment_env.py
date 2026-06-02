"""M3 OSP-Alignment-Lite Isaac environment.

This is a paper-bridge task, not a complete HALyPO reproduction. It creates an
orientation-sensitive pushing scene and exposes the paper's 11D command surface.
The first controller is a temporary root-stabilized arm tracker with contact
gating; replacing it with G1 WBC/Pink IK is the next fidelity step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from .osp_alignment_env_cfg import OSPAlignmentLiteEnvCfg


def _yaw_to_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((yaw.shape[0], 4), device=yaw.device)
    quat[:, 2] = torch.sin(0.5 * yaw)
    quat[:, 3] = torch.cos(0.5 * yaw)
    return quat


def _yaw_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


class OSPAlignmentLiteEnv(DirectRLEnv):
    """Direct Isaac Lab env for the first paper-like OSP alignment smoke."""

    cfg: OSPAlignmentLiteEnvCfg

    def __init__(self, cfg: OSPAlignmentLiteEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.command_11d = torch.zeros_like(self.actions)
        self.robot_local_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.robot_yaw = torch.zeros(self.num_envs, device=self.device)
        self.object_yaw = torch.zeros(self.num_envs, device=self.device)
        self.prev_object_x = torch.zeros(self.num_envs, device=self.device)
        self.left_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.partner_left_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.partner_right_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.contact_scores = torch.zeros((self.num_envs, 4), device=self.device)
        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.last_yaw_error = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.last_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_contact_scores = torch.zeros((self.num_envs, 4), device=self.device)
        self.obs_history = torch.zeros((self.num_envs, 3, self.cfg.core_observation_space), device=self.device)

        self._joint_ids = self._find_tracker_joints()

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.wall_left = RigidObject(self.cfg.wall_left_cfg)
        self.wall_right = RigidObject(self.cfg.wall_right_cfg)
        self.left_target = RigidObject(self.cfg.left_target_cfg)
        self.right_target = RigidObject(self.cfg.right_target_cfg)
        self.partner_left = RigidObject(self.cfg.partner_left_cfg)
        self.partner_right = RigidObject(self.cfg.partner_right_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        for name in (
            "object",
            "wall_left",
            "wall_right",
            "left_target",
            "right_target",
            "partner_left",
            "partner_right",
        ):
            self.scene.rigid_objects[name] = getattr(self, name)

        light_cfg = sim_utils.DomeLightCfg(intensity=2700.0, color=(0.78, 0.78, 0.76))
        light_cfg.func("/World/Light", light_cfg)

    def _find_tracker_joints(self) -> dict[str, int]:
        candidates = {
            "left_shoulder_pitch_joint": ("left_shoulder_pitch_joint",),
            "left_shoulder_roll_joint": ("left_shoulder_roll_joint",),
            "left_shoulder_yaw_joint": ("left_shoulder_yaw_joint",),
            "left_elbow_joint": ("left_elbow_joint", "left_elbow_pitch_joint"),
            "left_wrist_roll_joint": ("left_wrist_roll_joint", "left_elbow_roll_joint"),
            "right_shoulder_pitch_joint": ("right_shoulder_pitch_joint",),
            "right_shoulder_roll_joint": ("right_shoulder_roll_joint",),
            "right_shoulder_yaw_joint": ("right_shoulder_yaw_joint",),
            "right_elbow_joint": ("right_elbow_joint", "right_elbow_pitch_joint"),
            "right_wrist_roll_joint": ("right_wrist_roll_joint", "right_elbow_roll_joint"),
            "waist_pitch_joint": ("waist_pitch_joint", "torso_joint"),
        }
        ids: dict[str, int] = {}
        for name, patterns in candidates.items():
            for pattern in patterns:
                try:
                    joint_ids, _ = self.robot.find_joints(pattern)
                except ValueError:
                    continue
                if joint_ids:
                    ids[name] = joint_ids[0]
                    break
        return ids

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.command_11d[:, 0] = self.cfg.max_base_vx * self.actions[:, 0]
        self.command_11d[:, 1] = self.cfg.max_base_vy * self.actions[:, 1]
        self.command_11d[:, 2] = self.cfg.max_base_yaw_rate * self.actions[:, 2]
        self.command_11d[:, 3] = self.cfg.nominal_com_height + self.cfg.com_height_scale * self.actions[:, 3]
        self.command_11d[:, 4] = self.cfg.torso_pitch_scale * self.actions[:, 4]
        self.command_11d[:, 5:11] = self.cfg.wrist_delta_scale * self.actions[:, 5:11]

    def _apply_action(self) -> None:
        dt = self.cfg.sim.dt * self.cfg.decimation
        object_pos_before = self.object.data.root_pos_w.torch - self.scene.env_origins
        self.robot_local_xy[:, 0] += self.command_11d[:, 0] * dt
        self.robot_local_xy[:, 1] += self.command_11d[:, 1] * dt
        self.robot_local_xy[:, 0] = self.robot_local_xy[:, 0].clamp(min=-1.35, max=1.65)
        self.robot_local_xy[:, 0] = torch.minimum(self.robot_local_xy[:, 0], object_pos_before[:, 0] - 1.05)
        self.robot_local_xy[:, 1] = self.robot_local_xy[:, 1].clamp(min=-0.55, max=0.55)
        self.robot_yaw = wrap_to_pi(self.robot_yaw + self.command_11d[:, 2] * dt)

        object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        self.object_yaw = _yaw_from_quat_xyzw(self.object.data.root_quat_w.torch)
        contact_points = self._object_contact_points(object_pos, self.object_yaw)
        self._update_wrist_targets(contact_points)
        self._write_robot_tracker()
        self._write_marker(self.left_target, self.left_target_pos)
        self._write_marker(self.right_target, self.right_target_pos)
        self._write_marker(self.partner_left, self.partner_left_pos)
        self._write_marker(self.partner_right, self.partner_right_pos)
        self._apply_contact_gated_object_velocity(contact_points)

    def _object_contact_points(self, object_pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        local = torch.tensor(
            [
                [-0.5 * self.cfg.object_length - 0.04, 0.28 * self.cfg.object_width, 0.72],
                [-0.5 * self.cfg.object_length - 0.04, -0.28 * self.cfg.object_width, 0.72],
                [0.5 * self.cfg.object_length + 0.04, 0.25 * self.cfg.object_width, 0.72],
                [0.5 * self.cfg.object_length + 0.04, -0.25 * self.cfg.object_width, 0.72],
            ],
            device=self.device,
        )
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        rot = torch.stack(
            (
                torch.stack((c, -s), dim=-1),
                torch.stack((s, c), dim=-1),
            ),
            dim=-2,
        )
        points = local.unsqueeze(0).repeat(self.num_envs, 1, 1)
        points[:, :, 0:2] = torch.bmm(points[:, :, 0:2], rot.transpose(1, 2))
        points[:, :, 0:2] += object_pos[:, None, 0:2]
        points[:, :, 2] = object_pos[:, None, 2] + local[None, :, 2]
        return points + self.scene.env_origins[:, None, :]

    def _update_wrist_targets(self, contact_points: torch.Tensor) -> None:
        origins = self.scene.env_origins
        base = torch.cat(
            (
                self.robot_local_xy,
                self.command_11d[:, 3:4],
            ),
            dim=-1,
        ) + origins
        nominal_left = torch.tensor([0.58, 0.23, 0.10], device=self.device)
        nominal_right = torch.tensor([0.58, -0.23, 0.10], device=self.device)
        self.left_target_pos = base + nominal_left + self.command_11d[:, 5:8]
        self.right_target_pos = base + nominal_right + self.command_11d[:, 8:11]
        self.partner_left_pos = 0.85 * self.partner_left_pos + 0.15 * contact_points[:, 2]
        self.partner_right_pos = 0.85 * self.partner_right_pos + 0.15 * contact_points[:, 3]

    def _apply_contact_gated_object_velocity(self, contact_points: torch.Tensor) -> None:
        g1_left = torch.exp(-0.5 * (torch.linalg.norm(self.left_target_pos - contact_points[:, 0], dim=-1) / self.cfg.contact_sigma).square())
        g1_right = torch.exp(
            -0.5 * (torch.linalg.norm(self.right_target_pos - contact_points[:, 1], dim=-1) / self.cfg.contact_sigma).square()
        )
        partner_left = torch.exp(
            -0.5 * (torch.linalg.norm(self.partner_left_pos - contact_points[:, 2], dim=-1) / self.cfg.contact_sigma).square()
        )
        partner_right = torch.exp(
            -0.5 * (torch.linalg.norm(self.partner_right_pos - contact_points[:, 3], dim=-1) / self.cfg.contact_sigma).square()
        )
        self.contact_scores = torch.stack((g1_left, g1_right, partner_left, partner_right), dim=-1)
        g1_contact = 0.5 * (g1_left + g1_right)
        partner_contact = 0.5 * (partner_left + partner_right)
        coupled_contact = torch.minimum(g1_contact, partner_contact).clamp(0.0, 1.0)
        lateral_balance = (g1_left - g1_right).clamp(-1.0, 1.0)

        root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        root_vel[:, 0] = self.cfg.contact_push_scale * torch.relu(self.command_11d[:, 0]) * coupled_contact
        root_vel[:, 1] = 0.35 * self.cfg.contact_push_scale * self.command_11d[:, 1] * coupled_contact
        yaw_error = wrap_to_pi(self.cfg.goal_yaw - self.object_yaw)
        root_vel[:, 5] = (
            self.cfg.contact_yaw_scale * coupled_contact * yaw_error
            + 0.25 * self.cfg.contact_yaw_scale * lateral_balance
        )
        self.object.write_root_velocity_to_sim_index(root_velocity=root_vel)

    def _write_robot_tracker(self) -> None:
        env_ids = self.robot._ALL_INDICES
        origins = self.scene.env_origins
        root_pose = self.robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] += origins
        root_pose[:, 0] = origins[:, 0] + self.robot_local_xy[:, 0]
        root_pose[:, 1] = origins[:, 1] + self.robot_local_xy[:, 1]
        root_pose[:, 2] = self.command_11d[:, 3]
        root_pose[:, 3:7] = _yaw_to_quat_xyzw(self.robot_yaw)
        root_vel = torch.zeros_like(self.robot.data.default_root_vel.torch)

        joint_pos = self.robot.data.default_joint_pos.torch.clone()
        torso = self.command_11d[:, 4]
        left_delta = self.command_11d[:, 5:8]
        right_delta = self.command_11d[:, 8:11]
        self._set_joint(joint_pos, "waist_pitch_joint", torso)
        self._set_joint(joint_pos, "left_shoulder_pitch_joint", 0.45 - 0.8 * left_delta[:, 2])
        self._set_joint(joint_pos, "left_shoulder_roll_joint", 0.18 + 1.2 * left_delta[:, 1])
        self._set_joint(joint_pos, "left_shoulder_yaw_joint", 0.25 + 0.8 * left_delta[:, 0])
        self._set_joint(joint_pos, "left_elbow_joint", 0.75 + 0.6 * left_delta[:, 0])
        self._set_joint(joint_pos, "left_wrist_roll_joint", 0.3 * left_delta[:, 1])
        self._set_joint(joint_pos, "right_shoulder_pitch_joint", 0.45 - 0.8 * right_delta[:, 2])
        self._set_joint(joint_pos, "right_shoulder_roll_joint", -0.18 + 1.2 * right_delta[:, 1])
        self._set_joint(joint_pos, "right_shoulder_yaw_joint", -0.25 + 0.8 * right_delta[:, 0])
        self._set_joint(joint_pos, "right_elbow_joint", 0.75 + 0.6 * right_delta[:, 0])
        self._set_joint(joint_pos, "right_wrist_roll_joint", 0.3 * right_delta[:, 1])

        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel.torch)
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _set_joint(self, joint_pos: torch.Tensor, name: str, value: torch.Tensor) -> None:
        if name in self._joint_ids:
            joint_pos[:, self._joint_ids[name]] = value

    def _write_marker(self, marker: RigidObject, pos_w: torch.Tensor) -> None:
        root_pose = marker.data.default_root_pose.torch.clone()
        root_pose[:, :3] = pos_w
        marker.write_root_pose_to_sim_index(root_pose=root_pose)
        marker.write_root_velocity_to_sim_index(root_velocity=torch.zeros((self.num_envs, 6), device=self.device))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        core = self._compute_core_obs()
        self.obs_history[:, 2] = self.obs_history[:, 1].clone()
        self.obs_history[:, 1] = self.obs_history[:, 0].clone()
        self.obs_history[:, 0] = core
        return {"policy": self.obs_history.reshape(self.num_envs, -1)}

    def _compute_core_obs(self) -> torch.Tensor:
        object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        object_vel = self.object.data.root_lin_vel_w.torch
        object_yaw = _yaw_from_quat_xyzw(self.object.data.root_quat_w.torch)
        yaw_error = wrap_to_pi(self.cfg.goal_yaw - object_yaw)
        goal_delta = torch.stack((self.cfg.goal_x - object_pos[:, 0], -object_pos[:, 1], yaw_error), dim=-1)
        waypoints = torch.tensor([[0.65, 0.0], [1.35, 0.0], [self.cfg.goal_x, 0.0]], device=self.device)
        rel_waypoints = waypoints.unsqueeze(0).repeat(self.num_envs, 1, 1) - object_pos[:, None, 0:2]
        target_rel = torch.cat(
            (
                self.left_target_pos - self.scene.env_origins - object_pos,
                self.right_target_pos - self.scene.env_origins - object_pos,
                self.partner_left_pos - self.scene.env_origins - object_pos,
                self.partner_right_pos - self.scene.env_origins - object_pos,
            ),
            dim=-1,
        )
        core = torch.cat(
            (
                object_pos[:, 0:3],
                object_vel[:, 0:2],
                torch.cos(object_yaw).unsqueeze(-1),
                torch.sin(object_yaw).unsqueeze(-1),
                goal_delta,
                self.robot_local_xy,
                torch.cos(self.robot_yaw).unsqueeze(-1),
                torch.sin(self.robot_yaw).unsqueeze(-1),
                self.command_11d,
                self.contact_scores,
                target_rel,
                rel_waypoints.reshape(self.num_envs, -1),
                torch.full((self.num_envs, 1), 1.14, device=self.device),
            ),
            dim=-1,
        )
        if core.shape[1] < self.cfg.core_observation_space:
            core = torch.cat(
                (core, torch.zeros((self.num_envs, self.cfg.core_observation_space - core.shape[1]), device=self.device)),
                dim=-1,
            )
        return core[:, : self.cfg.core_observation_space]

    def _get_rewards(self) -> torch.Tensor:
        object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        object_yaw = _yaw_from_quat_xyzw(self.object.data.root_quat_w.torch)
        progress = object_pos[:, 0] - self.prev_object_x
        self.prev_object_x = object_pos[:, 0].clone()
        distance = torch.linalg.norm(torch.stack((self.cfg.goal_x - object_pos[:, 0], -object_pos[:, 1]), dim=-1), dim=-1)
        yaw_error = torch.abs(wrap_to_pi(self.cfg.goal_yaw - object_yaw))
        contact_loss = (self.contact_scores[:, :2].mean(dim=-1) < self.cfg.contact_loss_threshold).float()
        reward = 6.0 * progress - 0.30 * distance - 0.35 * yaw_error - 0.20 * contact_loss
        success = (distance < self.cfg.success_pos_tolerance) & (yaw_error < self.cfg.success_yaw_tolerance)
        reward += 3.0 * success.float()
        self.extras.setdefault("log", {})["Metrics/mean_distance"] = distance.mean().item()
        self.extras.setdefault("log", {})["Metrics/mean_yaw_error"] = yaw_error.mean().item()
        self.extras.setdefault("log", {})["Metrics/mean_contact"] = self.contact_scores[:, :2].mean().item()
        self.extras.setdefault("log", {})["Metrics/success_rate"] = success.float().mean().item()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pos = self.object.data.root_pos_w.torch - self.scene.env_origins
        object_yaw = _yaw_from_quat_xyzw(self.object.data.root_quat_w.torch)
        distance = torch.linalg.norm(torch.stack((self.cfg.goal_x - object_pos[:, 0], -object_pos[:, 1]), dim=-1), dim=-1)
        yaw_error = torch.abs(wrap_to_pi(self.cfg.goal_yaw - object_yaw))
        success = (distance < self.cfg.success_pos_tolerance) & (yaw_error < self.cfg.success_yaw_tolerance)
        failed = (torch.abs(object_pos[:, 1]) > self.cfg.max_lateral_drift) | (object_pos[:, 2] < self.cfg.min_object_height)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.last_success = success
        self.last_distance = distance
        self.last_yaw_error = yaw_error
        self.last_object_pos = object_pos.clone()
        self.last_contact_scores = self.contact_scores.clone()
        return success | failed, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.object._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids_t]

        object_pose = self.object.data.default_root_pose.torch[env_ids_t].clone()
        object_pose[:, :3] += origins
        object_pose[:, 0] = origins[:, 0] + self.cfg.object_start_x
        object_pose[:, 1] = origins[:, 1]
        object_pose[:, 2] = self.cfg.object_start_z
        start_yaw = torch.full((len(env_ids_t),), self.cfg.object_start_yaw, device=self.device)
        object_pose[:, 3:7] = _yaw_to_quat_xyzw(start_yaw)
        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids_t)
        self.object.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(env_ids_t), 6), device=self.device), env_ids=env_ids_t
        )

        self.robot_local_xy[env_ids_t, 0] = -1.15
        self.robot_local_xy[env_ids_t, 1] = 0.0
        self.robot_yaw[env_ids_t] = 0.0
        self.object_yaw[env_ids_t] = self.cfg.object_start_yaw
        self.prev_object_x[env_ids_t] = self.cfg.object_start_x
        self.actions[env_ids_t] = 0.0
        self.command_11d[env_ids_t] = 0.0
        self.command_11d[env_ids_t, 3] = self.cfg.nominal_com_height
        self.contact_scores[env_ids_t] = 0.0
        self.obs_history[env_ids_t] = 0.0

        object_pos = object_pose[:, :3] - origins
        contact_points = self._object_contact_points_for_envs(env_ids_t, object_pos, start_yaw)
        self.partner_left_pos[env_ids_t] = contact_points[:, 2]
        self.partner_right_pos[env_ids_t] = contact_points[:, 3]
        self.left_target_pos[env_ids_t] = contact_points[:, 0]
        self.right_target_pos[env_ids_t] = contact_points[:, 1]
        self._write_robot_tracker()
        self._write_marker(self.left_target, self.left_target_pos)
        self._write_marker(self.right_target, self.right_target_pos)
        self._write_marker(self.partner_left, self.partner_left_pos)
        self._write_marker(self.partner_right, self.partner_right_pos)

    def _object_contact_points_for_envs(self, env_ids: torch.Tensor, object_pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        origins = self.scene.env_origins[env_ids]
        local = torch.tensor(
            [
                [-0.5 * self.cfg.object_length - 0.04, 0.28 * self.cfg.object_width, 0.72],
                [-0.5 * self.cfg.object_length - 0.04, -0.28 * self.cfg.object_width, 0.72],
                [0.5 * self.cfg.object_length + 0.04, 0.25 * self.cfg.object_width, 0.72],
                [0.5 * self.cfg.object_length + 0.04, -0.25 * self.cfg.object_width, 0.72],
            ],
            device=self.device,
        )
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        rot = torch.stack((torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2)
        points = local.unsqueeze(0).repeat(len(env_ids), 1, 1)
        points[:, :, 0:2] = torch.bmm(points[:, :, 0:2], rot.transpose(1, 2))
        points[:, :, 0:2] += object_pos[:, None, 0:2] + origins[:, None, 0:2]
        points[:, :, 2] = object_pos[:, None, 2] + local[None, :, 2]
        return points
