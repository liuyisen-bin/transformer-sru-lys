"""Unitree G1 EDU 29-DOF navigation with Unitree's velocity policy."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import engineai_rl_lab.tasks.navigation.mdp as mdp
from engineai_rl_lab.tasks.navigation.assets.g1 import (
    G1_29DOF_CFG,
    G1_29DOF_JOINT_NAMES,
)
from engineai_rl_lab.tasks.navigation.g1.unitree_velocity_policy import (
    get_unitree_velocity_policy_path,
)
from engineai_rl_lab.tasks.navigation.mdp.navigation.actions.navigation_se2_actions_unitree import (
    UNITREE_POLICY_JOINT_NAMES,
)
from engineai_rl_lab.tasks.navigation.config.pm01.navigation_env_cfg import (
    NAVIGATION_DECIMATION,
    PM01_PHYSICS_MATERIAL,
    ActionsCfg as PM01ActionsCfg,
    CommandsCfg as PM01CommandsCfg,
    CurriculumCfg as PM01CurriculumCfg,
    EventCfg as PM01EventCfg,
    ObservationsCfg as PM01ObservationsCfg,
    PM01NavigationEnvCfg,
    PM01SceneCfg,
    RewardsCfg as PM01RewardsCfg,
    TerminationsCfg as PM01TerminationsCfg,
)
from engineai_rl_lab.tasks.navigation.mdp.delay_manager import ObservationDelayManagerCfg
from engineai_rl_lab.tasks.terrains import HfMazeTerrainCfg


SIM_DT = 0.005
SIM_FREQUENCY = 200.0
NAVIGATION_FREQUENCY = 5.0
LOW_LEVEL_DECIMATION = 4  # 200 Hz physics -> 50 Hz Unitree policy

# The encoder contract is a fixed, endpoint-inclusive 16 x 41 grid. Specify
# the number of samples directly instead of relying on simulator rounding of
# horizontal_res (which produced 42 columns on the training machine).
G1_LIDAR_CHANNELS = 16
G1_LIDAR_COLUMNS = 41
G1_LIDAR_HORIZONTAL_RESOLUTION = 120.0 / (G1_LIDAR_COLUMNS - 1)


def g1_lidar_pattern(cfg: patterns.LidarPatternCfg, device: str):
    """Return the exact encoder grid, ordered by elevation then azimuth."""
    import torch

    # Generate on CPU, as in the reference Isaac Lab pattern, then transfer.
    vertical = torch.deg2rad(torch.linspace(*cfg.vertical_fov_range, cfg.channels))
    horizontal = torch.deg2rad(torch.linspace(*cfg.horizontal_fov_range, G1_LIDAR_COLUMNS))
    elevation, azimuth = torch.meshgrid(vertical, horizontal, indexing="ij")
    directions = torch.stack(
        (
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.sin(elevation),
        ),
        dim=-1,
    ).reshape(-1, 3).to(device)
    return torch.zeros_like(directions), directions

G1_PHYSICS_MATERIAL = PM01_PHYSICS_MATERIAL


@configclass
class G1SceneCfg(PM01SceneCfg):
    """G1 asset, navigation LiDAR, height scan, contact sensor, and lighting."""

    robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Fixed sensor frames are merged, avoiding fictitious rigid-body masses.
    # Follow the torso's physics pose and retain the exact mid360_joint mount.
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        mesh_prim_paths=["/World/ground"],
        update_period=NAVIGATION_DECIMATION * SIM_DT,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0002835, 0.00003, 0.40618),
            rot=(0.9997985784932998, 0.0, 0.020069938783589012, 0.0),
        ),
        ray_alignment="base",
        debug_vis=True,
        max_distance=5.0,
        pattern_cfg=patterns.LidarPatternCfg(
            func=g1_lidar_pattern,
            channels=G1_LIDAR_CHANNELS,
            vertical_fov_range=(-45.0, 0.0),
            horizontal_fov_range=(-60.0, 60.0),
            horizontal_res=G1_LIDAR_HORIZONTAL_RESOLUTION,
        ),
    )

    height_scanner_critic = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[10.0, 10.0]),
        update_period=NAVIGATION_DECIMATION * SIM_DT,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=SIM_DT,
        history_length=NAVIGATION_DECIMATION,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/"
                "PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


@configclass
class G1ActionsCfg(PM01ActionsCfg):
    """Existing 3-D navigation action with Unitree's 29-DOF backend."""

    velocity_command = mdp.UnitreeNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_position_action=mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=UNITREE_POLICY_JOINT_NAMES,
            scale=0.25,
            use_default_offset=True,
            preserve_order=True,
        ),
        policy_file=str(get_unitree_velocity_policy_path(validate=False)),
        joint_names=UNITREE_POLICY_JOINT_NAMES,
        low_level_decimation=LOW_LEVEL_DECIMATION,
        policy_scaling=[1.0, 1.0, 1.0],
    )


@configclass
class G1ObservationsCfg(PM01ObservationsCfg):
    """Reuse PM01 navigation input layout and 16x41 LiDAR contract."""


@configclass
class G1EventCfg(PM01EventCfg):
    """G1 reset/event settings with a narrower initial joint perturbation."""

    # Retain the shared material randomization on non-foot bodies. The frozen
    # Unitree policy was validated with the foot material in assets/g1.py;
    # PM01's 0.8--1.2 foot friction would reintroduce the pure-right-turn stall.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!.*_ankle_roll_link$).*"]),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class G1RewardsCfg(PM01RewardsCfg):
    """Reuse PM01 navigation rewards over the G1 29-DOF articulation."""

    joint_acc_l2_joint = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=G1_29DOF_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )


@configclass
class G1TerminationsCfg(PM01TerminationsCfg):
    """G1 collision/fall names; goal and terrain conditions stay shared."""

    base_contact = DoneTerm(
        func=mdp.illegal_contact_navigation,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "pelvis",
                    "torso_link",
                    ".*_knee_link",
                    ".*_shoulder_.*_link",
                    ".*_elbow_link",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class G1NavigationEnvCfg(PM01NavigationEnvCfg):
    """Registered G1 task retaining the proven navigation environment."""

    scene: G1SceneCfg = G1SceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=False,
    )
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: PM01CommandsCfg = PM01CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventCfg = G1EventCfg()
    curriculum: PM01CurriculumCfg = PM01CurriculumCfg()
    delay_cfg: ObservationDelayManagerCfg = ObservationDelayManagerCfg(
        enabled=True,
        max_delay_lin_vel=2,
        max_delay_ang_vel=2,
        max_delay_projected_gravity=2,
        max_delay_target_position=2,
        max_delay_lidar=2,
    )

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=LOW_LEVEL_DECIMATION,
        physics_material=G1_PHYSICS_MATERIAL,
    )
    decimation = NAVIGATION_DECIMATION
    low_level_decimation = LOW_LEVEL_DECIMATION
    episode_length_s = 60.0
    is_finite_horizon = True

    def __post_init__(self):
        pattern_cfg = self.scene.ray_caster.pattern_cfg
        _, ray_directions = pattern_cfg.func(pattern_cfg, device="cpu")
        expected_rays = G1_LIDAR_CHANNELS * G1_LIDAR_COLUMNS
        actual_rays = ray_directions.shape[0]
        if actual_rays != expected_rays:
            actual_columns = actual_rays // G1_LIDAR_CHANNELS
            raise ValueError(
                "G1 LiDAR shape mismatch: expected "
                f"{G1_LIDAR_CHANNELS}x{G1_LIDAR_COLUMNS}={expected_rays} rays, "
                f"but the configured pattern generated {G1_LIDAR_CHANNELS}x{actual_columns}="
                f"{actual_rays}."
            )

        # Validate the bundled simulation asset and Unitree policy at task creation.
        if not os.path.isfile(self.scene.robot.spawn.asset_path):
            raise FileNotFoundError(
                f"Bundled G1 29-DOF simulation URDF not found: {self.scene.robot.spawn.asset_path}."
            )
        self.actions.velocity_command.policy_file = str(
            get_unitree_velocity_policy_path(validate=True)
        )


@configclass
class G1NavigationEnvCfg_PLAY(G1NavigationEnvCfg):
    """Small PLAY configuration for visual/chain smoke checks."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class G1NavigationEnvCfg_VISUAL(G1NavigationEnvCfg_PLAY):
    """Minimal flat scene for inspecting the initial G1 state on low-memory PCs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1

        # Keep the goal/spawn masks required by the navigation command, but
        # replace the 2x2 30 m terrain set with one obstacle-free 10 m patch.
        terrain = self.scene.terrain.terrain_generator
        terrain.size = (10.0, 10.0)
        terrain.border_width = 5.0
        terrain.num_rows = 1
        terrain.num_cols = 1
        terrain.curriculum = False
        terrain.difficulty_range = (0.0, 0.0)
        terrain.sub_terrains = {
            "visual_flat": HfMazeTerrainCfg(
                proportion=1.0,
                open_probability=1.0,
                grid_size=(5, 5),
                cell_size=2.0,
                wall_height=1.0,
                add_noise_to_flat=False,
                add_goal=True,
                randomize_wall=False,
                random_wall_ratio=0.0,
                non_maze_terrain=True,
                dynamic_obstacles=False,
            )
        }

        # Avoid debug geometry and large texture-backed materials in the
        # low-memory visualization path.  Training observations stay enabled.
        self.scene.ray_caster.debug_vis = False
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.18, 0.24, 0.18),
            roughness=1.0,
        )
        self.scene.sky_light.spawn.texture_file = None
        self.scene.sky_light.spawn.color = (0.8, 0.8, 0.8)


__all__ = [
    "G1NavigationEnvCfg",
    "G1NavigationEnvCfg_PLAY",
    "G1NavigationEnvCfg_VISUAL",
    "G1SceneCfg",
    "G1ActionsCfg",
]
