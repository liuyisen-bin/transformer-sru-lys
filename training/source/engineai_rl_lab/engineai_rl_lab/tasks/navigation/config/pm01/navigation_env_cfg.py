"""Standalone PM01 Edu navigation environment configuration."""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensorCfg,
    RayCasterCfg,
    patterns,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import engineai_rl_lab.tasks.navigation.mdp as mdp
from engineai_rl_lab.tasks.navigation.assets import (
    ISAACLAB_NAV_TASKS_ASSETS_DIR,
    PM01_ACTION_SCALE,
    PM01_ACTIVE_JOINT_NAMES,
    PM01_CYLINDER_CFG,
)
from engineai_rl_lab.tasks.navigation.mdp.custom_noise import (
    DeltaTransformationNoiseCfg,
)
from engineai_rl_lab.tasks.navigation.mdp.delay_manager import (
    ObservationDelayManagerCfg,
)
from engineai_rl_lab.tasks.terrains import HfMazeTerrainCfg


SIM_DT = 0.005
SIM_FREQUENCY = 200.0
NAVIGATION_FREQUENCY = 5.0
NAVIGATION_DECIMATION = int(SIM_FREQUENCY / NAVIGATION_FREQUENCY)
LOW_LEVEL_DECIMATION = 2


PM01_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    restitution=0.1,
    static_friction=1.0,
    dynamic_friction=0.8,
    compliant_contact_stiffness=5.0e5,
    compliant_contact_damping=300.0,
)


PM01_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(30.0, 30.0),
    border_width=30.0,
    num_rows=6,
    num_cols=30,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=False,
    difficulty_range=(0.5, 1.0),
    sub_terrains={
        "maze": HfMazeTerrainCfg(
            proportion=0.3,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=0.5,
            add_stairs_to_maze=False,
        ),
        # 1 m 高台不作为目标采样区域。
        "non_maze": HfMazeTerrainCfg(
            proportion=0.5,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            wall_height=1.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
        ),
        "pits": HfMazeTerrainCfg(
            proportion=0.2,
            open_probability=0.9,
            grid_size=(15, 15),
            cell_size=2.0,
            add_noise_to_flat=False,
            add_goal=True,
            randomize_wall=True,
            random_wall_ratio=1.0,
            non_maze_terrain=True,
            dynamic_obstacles=True,
        ),
    },
)


@configclass
class PM01SceneCfg(InteractiveSceneCfg):
    """PM01 robot, terrain, sensors, and lighting."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PM01_TERRAIN_GENERATOR_CFG,
        max_init_terrain_level=10,
        collision_group=-1,
        physics_material=PM01_PHYSICS_MATERIAL,
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot = PM01_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # yaw 对齐只旋转射线起点，不旋转方向；雷达需用 base 对齐。
    # 改雷达参数时同步 range_image_rsl_rl_cfg.py。Isaac Lab 2.3 使用
    # ceil((h_max-h_min)/horizontal_res) 计算水平采样数；这里保持训练时的 41 列。
    # range_image_shape=(channels, N_h)，range_image_max_distance=max_distance。
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LINK_TORSO_YAW",
        mesh_prim_paths=["/World/ground"],
        update_period=NAVIGATION_DECIMATION * SIM_DT,
        offset=RayCasterCfg.OffsetCfg(pos=(0.105, 0.0, 0.185)),
        ray_alignment="base",
        debug_vis=True,
        max_distance=5.0,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=(-45.0, 0.0),
            horizontal_fov_range=(-60.0, 60.0),
            horizontal_res=120.0 / 41.0,
        ),
    )

    height_scanner_critic = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LINK_BASE",
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
class CommandsCfg:
    """Navigation goal command configuration."""

    robot_goal = mdp.RobotNavigationGoalCommandCfg(
        asset_name="robot",
        # 极大周期关闭自动重采样，目标只在回合重置时改变。
        resampling_time_range=(1.0e9, 1.0e9),
        # 到达目标后需连续停留的时间。
        required_time_at_goal_s=1.0,
        robot_to_goal_line_vis=True,
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """PM01 hierarchical navigation action."""

    velocity_command = mdp.HumanoidNavigationSE2ActionCfg(
        asset_name="robot",
        low_level_position_action=mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=PM01_ACTIVE_JOINT_NAMES,
            scale=PM01_ACTION_SCALE,
            use_default_offset=True,
            preserve_order=True,
        ),
        active_joint_names=PM01_ACTIVE_JOINT_NAMES,
        low_level_decimation=LOW_LEVEL_DECIMATION,
        low_level_policy_file=os.path.join(
            ISAACLAB_NAV_TASKS_ASSETS_DIR,
            "Policies",
            "locomotion",
            "pm01",
            "251009_221907_41000.pt",
        ),
        policy_scaling=[0.7, 0.25, 0.7],
        use_raw_actions=True,
        policy_distr_type="gaussian",
    )


@configclass
class ObservationsCfg:
    """Actor, critic, low-level, and metric observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel_delayed,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_delayed,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_delayed,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        last_action = ObsTerm(func=mdp.last_action)
        target_position = ObsTerm(
            func=mdp.generated_commands_reshaped_delayed,
            params={"command_name": "robot_goal", "flatten": True},
            noise=DeltaTransformationNoiseCfg(
                rotation=0.1,
                translation=0.5,
                noise_prob=0.1,
                remove_dist=False,
            ),
        )
        lidar_point_cloud = ObsTerm(
            func=mdp.lidar_point_cloud_delayed,
            params={"sensor_cfg": SceneEntityCfg("ray_caster")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        last_action = ObsTerm(func=mdp.last_action)
        target_position = ObsTerm(
            func=mdp.generated_commands_reshaped,
            params={"command_name": "robot_goal", "flatten": True},
        )
        time_normalized = ObsTerm(
            func=mdp.time_normalized,
            params={"command_name": "robot_goal"},
        )
        height_scan_critic = ObsTerm(
            func=mdp.height_scan_feat,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_critic")},
        )
        lidar_point_cloud = ObsTerm(
            func=mdp.lidar_point_cloud,
            params={"sensor_cfg": SceneEntityCfg("ray_caster")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LowLevelPolicyCfg(ObsGroup):
        """Diagnostics only; the humanoid action builds its own history."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_actions,
            params={"action_name": "velocity_command"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class MetricsCfg(ObsGroup):
        in_goal = ObsTerm(func=mdp.in_goal)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    metrics: MetricsCfg = MetricsCfg()
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    low_level_policy: LowLevelPolicyCfg = LowLevelPolicyCfg()


@configclass
class EventCfg:
    """Startup, reset, and interval randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_action_scale = EventTerm(
        func=mdp.randomize_action_scale,
        mode="reset",
        params={
            "scale_range_x": (0.8, 1.2),
            "scale_range_y": (0.6, 1.0),
            "scale_range_theta": (0.8, 1.2),
            "scale_range_xb": 0.1,
            "scale_range_yb": 0.2,
            "scale_range_thetab": 0.1,
            "action_term": "velocity_command",
        },
    )

    reset_delay_buffer = EventTerm(
        func=mdp.reset_and_randomize_delay_buffer,
        mode="reset",
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """PM01 navigation rewards."""

    joint_acc_l2_joint = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=PM01_ACTIVE_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    lateral_movement = RewTerm(func=mdp.lateral_movement, weight=-0.1)
    rot_movement = RewTerm(func=mdp.rot_movement, weight=-1.0e-5)
    action_rate_l1 = RewTerm(func=mdp.action_rate_l1, weight=-0.1)
    episode_termination = RewTerm(func=mdp.is_terminated, weight=-50.0)

    reach_goal_xy_soft = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=0.25,
        params={
            "command_name": "robot_goal",
            "sigmoid": 2.5,
            "T_r": 1.0,
            "probability": 0.01,
            "flat": False,
            "ratio": False,
        },
    )
    reach_goal_xy_tight = RewTerm(
        func=mdp.reach_goal_xyz,
        weight=1.5,
        params={
            "command_name": "robot_goal",
            "sigmoid": 0.25,
            "T_r": 0.1,
            "probability": 0.01,
            "flat": True,
            "ratio": False,
        },
    )
    backward_movement_penalty = RewTerm(
        func=mdp.backward_movement_penalty,
        weight=-0.0,
    )


@configclass
class TerminationsCfg:
    """PM01 navigation termination conditions."""

    time_out = DoneTerm(
        func=mdp.time_out_navigation,
        time_out=True,
        params={"distance_threshold": 0.5},
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact_navigation,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "LINK_BASE",
                    ".*KNEE.*",
                    ".*SHOULDER.*",
                    ".*ELBOW.*",
                    "LINK_TORSO_YAW",
                ],
            ),
            "threshold": 1.0,
        },
    )
    large_pitch_angle = DoneTerm(
        func=mdp.large_angle_termination_navigation,
        params={"threshold": 40},
    )
    early_termination = DoneTerm(
        func=mdp.at_goal_navigation,
        time_out=True,
        params={"distance_threshold": 0.5},
    )
    terrain_fall = DoneTerm(
        func=mdp.terrain_fall,
        params={"fall_height_threshold": -2.0},
    )


@configclass
class CurriculumCfg:
    """Navigation curriculum."""

    disable_backward_penalty = CurrTerm(
        func=mdp.disable_backward_penalty_after_steps,
        params={
            "disable_after_steps": 500,
            "action_term": "velocity_command",
        },
    )


@configclass
class PM01NavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Standalone PM01 navigation task configuration."""

    scene: PM01SceneCfg = PM01SceneCfg(
        num_envs=2048,
        env_spacing=2.5,
        replicate_physics=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
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
        physics_material=PM01_PHYSICS_MATERIAL,
    )
    decimation = NAVIGATION_DECIMATION
    low_level_decimation = LOW_LEVEL_DECIMATION
    episode_length_s = 60.0
    is_finite_horizon = True

    def __post_init__(self):
        pass


@configclass
class PM01NavigationEnvCfg_PLAY(PM01NavigationEnvCfg):
    """PM01 play configuration with fewer environments and no policy corruption."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 20
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
