"""Configuration for the M3 OSP-Alignment-Lite Isaac task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from isaaclab_assets import G1_MINIMAL_CFG


G1_OSP_TRACKER_CFG = G1_MINIMAL_CFG.copy()
G1_OSP_TRACKER_CFG.spawn.rigid_props.disable_gravity = True
G1_OSP_TRACKER_CFG.init_state.pos = (-1.15, 0.0, 0.78)


def _wall_cfg(name: str, y: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(1.35, 0.14, 0.55),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=9999.0,
                angular_damping=9999.0,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1_000_000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.46, 0.48, 0.50)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.45, y, 0.275)),
    )


def _marker_cfg(name: str, color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.SphereCfg(
            radius=0.055,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.9)),
    )


@configclass
class OSPAlignmentLiteEnvCfg(DirectRLEnvCfg):
    """Paper-bridge scene for one OSP alignment scenario.

    The task exposes the paper's 11D command surface and OSP geometry, but the
    first tracker is deliberately temporary: G1 root pose is stabilized and arm
    joints are driven by a heuristic PD target while object motion is gated by
    wrist/contact scores. This keeps the controller dependency visible.
    """

    decimation = 30
    episode_length_s = 20.0
    action_space = 11
    observation_space = 210
    state_space = 0
    core_observation_space = 70

    object_length = 1.25
    object_width = 0.45
    object_height = 0.24
    object_mass = 20.0
    object_start_yaw = 0.28
    object_start_x = 0.0
    object_start_z = 0.14
    goal_x = 2.15
    goal_yaw = 0.0
    success_pos_tolerance = 0.18
    success_yaw_tolerance = 0.12
    max_lateral_drift = 0.75
    min_object_height = 0.08

    max_base_vx = 0.55
    max_base_vy = 0.30
    max_base_yaw_rate = 0.8
    wrist_delta_scale = 0.28
    nominal_com_height = 0.78
    com_height_scale = 0.10
    torso_pitch_scale = 0.25
    contact_distance = 0.18
    contact_sigma = 0.24
    contact_push_scale = 0.70
    contact_yaw_scale = 0.65
    contact_loss_threshold = 0.18

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=5.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot_cfg: ArticulationCfg = G1_OSP_TRACKER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(object_length, object_width, object_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=3.0,
                max_angular_velocity=3.0,
                linear_damping=1.8,
                angular_damping=1.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=object_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.42, 0.76)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(object_start_x, 0.0, object_start_z)),
    )
    wall_left_cfg: RigidObjectCfg = _wall_cfg("GateWallLeft", 0.57)
    wall_right_cfg: RigidObjectCfg = _wall_cfg("GateWallRight", -0.57)
    left_target_cfg: RigidObjectCfg = _marker_cfg("G1LeftWristTarget", (0.95, 0.25, 0.12))
    right_target_cfg: RigidObjectCfg = _marker_cfg("G1RightWristTarget", (0.95, 0.68, 0.12))
    partner_left_cfg: RigidObjectCfg = _marker_cfg("PartnerLeftWristProxy", (0.15, 0.70, 0.95))
    partner_right_cfg: RigidObjectCfg = _marker_cfg("PartnerRightWristProxy", (0.20, 0.90, 0.65))
