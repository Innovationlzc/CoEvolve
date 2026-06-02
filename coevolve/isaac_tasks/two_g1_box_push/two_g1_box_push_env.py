"""M2 Two-G1 Box Push Lite environment.

This milestone supports two action adapters. ``proxy_box_velocity`` directly
writes the box velocity. ``base_command`` first moves the two G1 base visuals and
only pushes the box through a simple virtual contact zone. The latter is still
not full locomotion/WBC, but it is the next step away from direct box control.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

if TYPE_CHECKING:
    from .two_g1_box_push_env_cfg import TwoG1BoxPushEnvCfg


class TwoG1BoxPushEnv(DirectRLEnv):
    """Direct Isaac Lab env for the first Two-G1 Box Push Lite smoke."""

    cfg: TwoG1BoxPushEnvCfg

    def __init__(self, cfg: TwoG1BoxPushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.goal_pos = torch.tensor([self.cfg.goal_x, 0.0, 0.2], device=self.device).repeat(self.num_envs, 1)
        self.prev_box_x = torch.zeros(self.num_envs, device=self.device)
        self.last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_box_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.last_robot_a_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_robot_b_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.robot_a_local_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.robot_b_local_xy = torch.zeros((self.num_envs, 2), device=self.device)

    def _setup_scene(self) -> None:
        self.robot_a = Articulation(self.cfg.robot_a_cfg)
        self.robot_b = Articulation(self.cfg.robot_b_cfg)
        self.box = RigidObject(self.cfg.box_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot_a"] = self.robot_a
        self.scene.articulations["robot_b"] = self.robot_b
        self.scene.rigid_objects["box"] = self.box

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.cfg.action_scale * actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        if self.cfg.action_mode == "proxy_box_velocity":
            self._apply_proxy_box_velocity()
        elif self.cfg.action_mode == "base_command":
            self._apply_base_command()
        else:
            raise ValueError(f"unknown M2 action_mode: {self.cfg.action_mode}")

    def _apply_proxy_box_velocity(self) -> None:
        """Average both agents' planar pushes and write box velocity directly."""

        push_a = self.actions[:, 0:2]
        push_b = self.actions[:, 2:4]
        mean_push = 0.5 * (push_a + push_b)
        mismatch = torch.linalg.norm(push_a - push_b, dim=-1, keepdim=True)
        damping = (1.0 - 0.25 * mismatch).clamp(min=0.25)
        root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        root_vel[:, 0:2] = 0.6 * mean_push * damping
        self.box.write_root_velocity_to_sim_index(root_velocity=root_vel)
        if self.cfg.keep_visual_robots_upright:
            self._sync_visual_robots()

    def _apply_base_command(self) -> None:
        """Move G1 base visuals first, then push the box through virtual contact."""

        dt = self.cfg.sim.dt * self.cfg.decimation
        command_a = self.actions[:, 0:2]
        command_b = self.actions[:, 2:4]
        self.robot_a_local_xy += self.cfg.base_command_speed * dt * command_a
        self.robot_b_local_xy += self.cfg.base_command_speed * dt * command_b

        box_pos = self.box.data.root_pos_w.torch - self.scene.env_origins
        max_robot_x = box_pos[:, 0] - self.cfg.base_command_x_standoff
        for robot_xy in (self.robot_a_local_xy, self.robot_b_local_xy):
            robot_xy[:, 0] = robot_xy[:, 0].clamp(min=self.cfg.base_command_x_min)
            robot_xy[:, 0] = torch.minimum(robot_xy[:, 0], max_robot_x)
            robot_xy[:, 1] = robot_xy[:, 1].clamp(
                min=-self.cfg.base_command_y_limit,
                max=self.cfg.base_command_y_limit,
            )

        contact_a = self._virtual_contact_score(self.robot_a_local_xy, box_pos, side=-1.0)
        contact_b = self._virtual_contact_score(self.robot_b_local_xy, box_pos, side=1.0)
        forward_push = torch.relu(command_a[:, 0]) * contact_a + torch.relu(command_b[:, 0]) * contact_b
        lateral_push = command_a[:, 1] * contact_a + command_b[:, 1] * contact_b
        contact_count = (contact_a + contact_b).clamp(min=1.0)

        root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        root_vel[:, 0] = self.cfg.virtual_contact_push_scale * forward_push / contact_count
        root_vel[:, 1] = 0.25 * self.cfg.virtual_contact_push_scale * lateral_push / contact_count
        self.box.write_root_velocity_to_sim_index(root_velocity=root_vel)
        if self.cfg.keep_visual_robots_upright:
            self._write_visual_robot_pose_from_xy(self.robot_a, self.robot_a_local_xy)
            self._write_visual_robot_pose_from_xy(self.robot_b, self.robot_b_local_xy)

    def _virtual_contact_score(self, robot_xy: torch.Tensor, box_pos: torch.Tensor, *, side: float) -> torch.Tensor:
        dx = box_pos[:, 0] - robot_xy[:, 0]
        dy = robot_xy[:, 1] - (box_pos[:, 1] + side * self.cfg.robot_y_offset)
        standoff_error = (dx - self.cfg.virtual_contact_standoff) / self.cfg.virtual_contact_sigma
        lateral_error = dy / self.cfg.virtual_contact_y_sigma
        behind_gate = (dx > 0.0).float()
        return behind_gate * torch.exp(-0.5 * (standoff_error.square() + lateral_error.square()))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        box_pos = self.box.data.root_pos_w.torch - self.scene.env_origins
        box_vel = self.box.data.root_lin_vel_w.torch
        robot_a_pos = self.robot_a.data.root_pos_w.torch - self.scene.env_origins
        robot_b_pos = self.robot_b.data.root_pos_w.torch - self.scene.env_origins
        rel_goal = self.goal_pos - box_pos
        obs = torch.cat(
            (
                box_pos[:, 0:2],
                box_vel[:, 0:2],
                rel_goal[:, 0:2],
                robot_a_pos[:, 0:2],
                robot_b_pos[:, 0:2],
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        box_pos = self.box.data.root_pos_w.torch - self.scene.env_origins
        distance = torch.linalg.norm(self.goal_pos[:, 0:2] - box_pos[:, 0:2], dim=-1)
        progress = box_pos[:, 0] - self.prev_box_x
        self.prev_box_x = box_pos[:, 0].clone()
        mismatch = torch.linalg.norm(self.actions[:, 0:2] - self.actions[:, 2:4], dim=-1)
        reward = 2.0 * progress - 0.1 * distance - 0.02 * mismatch
        reward += (distance < self.cfg.success_tolerance).float()
        self.extras.setdefault("log", {})["Metrics/mean_distance"] = distance.mean().item()
        self.extras.setdefault("log", {})["Metrics/success_rate"] = (distance < self.cfg.success_tolerance).float().mean().item()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        box_pos = self.box.data.root_pos_w.torch - self.scene.env_origins
        distance = torch.linalg.norm(self.goal_pos[:, 0:2] - box_pos[:, 0:2], dim=-1)
        reached = distance < self.cfg.success_tolerance
        out_of_bounds = torch.abs(box_pos[:, 1]) > 2.0
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.last_box_pos = box_pos.clone()
        self.last_distance = distance.clone()
        self.last_robot_a_pos = self.robot_a.data.root_pos_w.torch[:, :3] - self.scene.env_origins
        self.last_robot_b_pos = self.robot_b.data.root_pos_w.torch[:, :3] - self.scene.env_origins
        self.last_success = reached
        return reached | out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.box._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids_t]

        self._reset_robot(self.robot_a, env_ids_t, origins, y_offset=-0.65)
        self._reset_robot(self.robot_b, env_ids_t, origins, y_offset=0.65)

        box_pose = self.box.data.default_root_pose.torch[env_ids_t].clone()
        box_pose[:, :3] += origins
        box_pose[:, 0] += 0.0
        box_pose[:, 1] += 0.0
        box_vel = torch.zeros((len(env_ids_t), 6), device=self.device)
        self.box.write_root_pose_to_sim_index(root_pose=box_pose, env_ids=env_ids_t)
        self.box.write_root_velocity_to_sim_index(root_velocity=box_vel, env_ids=env_ids_t)

        self.actions[env_ids_t] = 0.0
        self.prev_box_x[env_ids_t] = 0.0
        self.robot_a_local_xy[env_ids_t, 0] = self.cfg.robot_x_offset
        self.robot_a_local_xy[env_ids_t, 1] = -self.cfg.robot_y_offset
        self.robot_b_local_xy[env_ids_t, 0] = self.cfg.robot_x_offset
        self.robot_b_local_xy[env_ids_t, 1] = self.cfg.robot_y_offset
        if self.cfg.keep_visual_robots_upright:
            if self.cfg.action_mode == "base_command":
                self._write_visual_robot_pose_from_xy(self.robot_a, self.robot_a_local_xy, env_ids_t)
                self._write_visual_robot_pose_from_xy(self.robot_b, self.robot_b_local_xy, env_ids_t)
            else:
                self._sync_visual_robots(env_ids_t)

    def _reset_robot(self, robot: Articulation, env_ids: torch.Tensor, origins: torch.Tensor, *, y_offset: float) -> None:
        root_pose = robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += origins
        root_pose[:, 0] += -0.75
        root_pose[:, 1] += y_offset
        root_vel = robot.data.default_root_vel.torch[env_ids].clone()
        joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = robot.data.default_joint_vel.torch[env_ids].clone()
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _write_visual_robot_pose_from_xy(
        self,
        robot: Articulation,
        local_xy: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = self.box._ALL_INDICES
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids_t]
        self._write_visual_robot_pose(
            robot,
            env_ids_t,
            origins,
            x_offset=local_xy[env_ids_t, 0],
            y_offset=local_xy[env_ids_t, 1],
        )

    def _sync_visual_robots(self, env_ids: torch.Tensor | None = None) -> None:
        """Keep the two G1 bodies upright while M2 still uses proxy box actions."""

        if env_ids is None:
            env_ids = self.box._ALL_INDICES
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.scene.env_origins[env_ids_t]
        box_pos = self.box.data.root_pos_w.torch[env_ids_t]
        anchor_x = box_pos[:, 0] - origins[:, 0] if self.cfg.robot_follow_box else torch.zeros_like(box_pos[:, 0])
        self._write_visual_robot_pose(
            self.robot_a,
            env_ids_t,
            origins,
            x_offset=anchor_x + self.cfg.robot_x_offset,
            y_offset=-self.cfg.robot_y_offset,
        )
        self._write_visual_robot_pose(
            self.robot_b,
            env_ids_t,
            origins,
            x_offset=anchor_x + self.cfg.robot_x_offset,
            y_offset=self.cfg.robot_y_offset,
        )

    def _write_visual_robot_pose(
        self,
        robot: Articulation,
        env_ids: torch.Tensor,
        origins: torch.Tensor,
        *,
        x_offset: torch.Tensor,
        y_offset: float,
    ) -> None:
        root_pose = robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += origins
        root_pose[:, 0] += x_offset
        root_pose[:, 1] += y_offset
        root_vel = torch.zeros_like(robot.data.default_root_vel.torch[env_ids])
        joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel.torch[env_ids])
        robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
