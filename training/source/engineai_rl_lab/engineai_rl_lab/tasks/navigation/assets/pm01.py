"""PM01 Edu articulation and locomotion-policy parameters."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PM01_ACTIVE_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
]

# Policy output-to-position scales from pm01_edu/rl_walking_example/default.yaml.
PM01_ACTION_SCALE = {
    ".*HIP_PITCH.*": 0.5,
    ".*HIP_ROLL.*": 0.2,
    ".*HIP_YAW.*": 0.2,
    ".*KNEE_PITCH.*": 0.5,
    ".*ANKLE_PITCH.*": 0.5,
    ".*ANKLE_ROLL.*": 0.2,
    ".*SHOULDER_PITCH.*": 0.2,
    ".*SHOULDER_ROLL.*": 0.2,
    ".*SHOULDER_YAW.*": 0.05,
    ".*ELBOW_PITCH.*": 0.2,
    ".*ELBOW_YAW.*": 0.05,
}


@configclass
class RobotArticulationCfg(ArticulationCfg):
    """Articulation config carrying the hardware SDK joint order."""

    joint_sdk_names: list[str] | None = None


PM01_CYLINDER_CFG = RobotArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(_ASSETS_DIR, "Robots", "pm01", "serial_pm01_edu.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*HIP_PITCH.*": -0.06,
            ".*KNEE_PITCH.*": 0.12,
            ".*ANKLE_PITCH.*": -0.06,
            "J14_SHOULDER_ROLL_L": 0.15,
            "J19_SHOULDER_ROLL_R": -0.15,
            ".*ELBOW_PITCH.*": -0.25,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim={
                ".*HIP_PITCH.*": 164.0,
                ".*HIP_ROLL.*": 164.0,
                ".*HIP_YAW.*": 61.0,
                ".*KNEE_PITCH.*": 164.0,
                ".*ANKLE.*": 54.9,
                ".*WAIST.*": 61.0,
                ".*SHOULDER.*": 61.0,
                ".*ELBOW.*": 61.0,
                ".*HEAD.*": 61.0,
            },
            velocity_limit_sim={
                ".*HIP_PITCH.*": 26.3,
                ".*HIP_ROLL.*": 26.3,
                ".*HIP_YAW.*": 35.2,
                ".*KNEE_PITCH.*": 26.3,
                ".*ANKLE.*": 35.2,
                ".*WAIST.*": 35.2,
                ".*SHOULDER.*": 35.2,
                ".*ELBOW.*": 35.2,
                ".*HEAD.*": 35.2,
            },
            stiffness={
                ".*HIP_PITCH.*": 110.0,
                ".*HIP_ROLL.*": 70.0,
                ".*HIP_YAW.*": 70.0,
                ".*KNEE_PITCH.*": 110.0,
                ".*ANKLE.*": 30.0,
                ".*WAIST.*": 80.0,
                ".*SHOULDER.*": 30.0,
                ".*ELBOW.*": 30.0,
                ".*HEAD.*": 50.0,
            },
            damping={
                ".*HIP_PITCH.*": 5.0,
                ".*HIP_ROLL.*": 3.0,
                ".*HIP_YAW.*": 3.0,
                ".*KNEE_PITCH.*": 5.0,
                ".*ANKLE.*": 0.3,
                ".*WAIST.*": 5.0,
                ".*SHOULDER.*": 0.3,
                ".*ELBOW.*": 0.3,
                ".*HEAD.*": 0.3,
            },
            armature={
                ".*HIP_PITCH.*": 0.0453,
                ".*HIP_ROLL.*": 0.0453,
                ".*KNEE_PITCH.*": 0.0453,
                ".*HIP_YAW.*": 0.0067,
                ".*ANKLE.*": 0.0067,
                ".*WAIST.*": 0.0067,
                ".*SHOULDER.*": 0.0067,
                ".*ELBOW.*": 0.0067,
                ".*HEAD.*": 0.0067,
            },
        )
    },
    joint_sdk_names=[
        "J00_HIP_PITCH_L",
        "J01_HIP_ROLL_L",
        "J02_HIP_YAW_L",
        "J03_KNEE_PITCH_L",
        "J04_ANKLE_PITCH_L",
        "J05_ANKLE_ROLL_L",
        "J06_HIP_PITCH_R",
        "J07_HIP_ROLL_R",
        "J08_HIP_YAW_R",
        "J09_KNEE_PITCH_R",
        "J10_ANKLE_PITCH_R",
        "J11_ANKLE_ROLL_R",
        "J12_WAIST_YAW",
        "J13_SHOULDER_PITCH_L",
        "J14_SHOULDER_ROLL_L",
        "J15_SHOULDER_YAW_L",
        "J16_ELBOW_PITCH_L",
        "J17_ELBOW_YAW_L",
        "J18_SHOULDER_PITCH_R",
        "J19_SHOULDER_ROLL_R",
        "J20_SHOULDER_YAW_R",
        "J21_ELBOW_PITCH_R",
        "J22_ELBOW_YAW_R",
        "J23_HEAD_YAW",
    ],
)

__all__ = [
    "PM01_ACTION_SCALE",
    "PM01_ACTIVE_JOINT_NAMES",
    "PM01_CYLINDER_CFG",
    "RobotArticulationCfg",
]
