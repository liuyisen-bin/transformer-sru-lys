"""Unitree G1 EDU 29-DOF articulation used by the navigation task."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

G1_29DOF_ASSET_DIR = Path(__file__).resolve().parent / "data" / "Robots" / "g1_29dof"
G1_29DOF_URDF = str(G1_29DOF_ASSET_DIR / "g1_29dof.urdf")

# Calibrated for this frozen policy in PhysX; this is not a measured shoe
# coefficient or a numerical copy of MuJoCo's soft friction constraints.
G1_FOOT_FRICTION = 0.3
# Ground compliance is configured by the environment. Do not add a second
# compliant material here: that changes the pair response (including turns).
G1_FOOT_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=G1_FOOT_FRICTION,
    dynamic_friction=G1_FOOT_FRICTION,
    restitution=0.1,
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
)


@sim_utils.clone
def spawn_g1_with_foot_material(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Bind the calibrated material before cloning the imported G1 asset."""
    from pxr import UsdShade

    prim = sim_utils.spawn_from_urdf(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    stage = prim.GetStage()
    material_path = f"{prim_path}/foot_physics_material"
    G1_FOOT_PHYSICS_MATERIAL.func(material_path, G1_FOOT_PHYSICS_MATERIAL)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        foot = stage.GetPrimAtPath(f"{prim_path}/{name}")
        if not foot.IsValid():
            raise RuntimeError(f"G1 foot link missing: {prim_path}/{name}")
        # Bind on the link so its instanceable collision children inherit it.
        UsdShade.MaterialBindingAPI.Apply(foot).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
    return prim

# Joint order of the bundled G1 29-DOF simulation asset.
G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1_29DOF_WBC_JOINT_NAMES = G1_29DOF_JOINT_NAMES[:15]
G1_29DOF_ARM_JOINT_NAMES = G1_29DOF_JOINT_NAMES[15:]

# Match Unitree velocity/v0 deploy.yaml.  The policy output is a residual around
# this pose, so these offsets are part of the learned low-level contract.
G1_29DOF_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_ankle_roll_joint": 0.0,
    "waist_.*_joint": 0.0,
    ".*_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
}


@configclass
class G1ArticulationCfg(ArticulationCfg):
    """Articulation config carrying the G1 joint-order contract."""

    gr00t_joint_names: list[str] | None = None


G1_29DOF_CFG = G1ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_g1_with_foot_material,
        asset_path=G1_29DOF_URDF,
        fix_base=False,
        # Empty sensor frames otherwise become three spurious 1 kg bodies.
        merge_fixed_joints=True,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        # Keep the conversion-time URDF drives neutral.  Isaac Lab applies the
        # actual PD gains below through the implicit articulation actuators.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos=G1_29DOF_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "unitree_legs": ImplicitActuatorCfg(
            # Match the official G1 motor inertia; the URDF importer defaults to zero.
            armature=0.01,
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                ".*_hip_.*_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*_ankle_.*_joint": 50.0,
                "waist_yaw_joint": 88.0,
                "waist_(roll|pitch)_joint": 50.0,
            },
            velocity_limit_sim={
                ".*_hip_.*_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*_ankle_.*_joint": 37.0,
                "waist_.*_joint": 32.0,
            },
            stiffness={
                ".*_hip_.*_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*_ankle_.*_joint": 40.0,
                "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_.*_joint": 2.0,
                ".*_knee_joint": 4.0,
                ".*_ankle_.*_joint": 2.0,
                "waist_.*_joint": 5.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            armature=0.01,
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_.*_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_(pitch|yaw)_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_.*_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_(pitch|yaw)_joint": 22.0,
            },
            stiffness=40.0,
            damping=10.0,
        ),
    },
    gr00t_joint_names=G1_29DOF_JOINT_NAMES,
)


__all__ = [
    "G1ArticulationCfg",
    "G1_29DOF_CFG",
    "G1_29DOF_URDF",
    "G1_29DOF_JOINT_NAMES",
    "G1_29DOF_WBC_JOINT_NAMES",
    "G1_29DOF_ARM_JOINT_NAMES",
    "G1_29DOF_DEFAULT_JOINT_POS",
    "G1_29DOF_ASSET_DIR",
]
