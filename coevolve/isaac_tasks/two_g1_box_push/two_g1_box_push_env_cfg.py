"""Configuration for the M2 Two-G1 Box Push Lite Isaac task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from isaaclab_assets import G1_MINIMAL_CFG


@configclass
class TwoG1BoxPushEnvCfg(DirectRLEnvCfg):
    """Lite validation scene: two G1 actors and one pushable box."""

    decimation = 2
    episode_length_s = 12.0
    action_space = 4
    observation_space = 12
    state_space = 0
    action_scale = 1.0
    goal_x = 2.0
    success_tolerance = 0.2
    action_mode = "proxy_box_velocity"
    keep_visual_robots_upright = True
    robot_follow_box = True
    robot_x_offset = -0.95
    robot_y_offset = 0.65
    base_command_speed = 0.75
    base_command_x_min = -1.4
    base_command_x_standoff = 0.75
    base_command_y_limit = 1.0
    virtual_contact_standoff = 0.8
    virtual_contact_sigma = 0.25
    virtual_contact_y_sigma = 0.35
    virtual_contact_push_scale = 0.75

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=5.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot_a_cfg: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/RobotA")
    robot_b_cfg: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/RobotB")
    box_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.45, 0.35),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=5.0,
                max_angular_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
    )
