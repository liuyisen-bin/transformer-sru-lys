# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Observation functions for navigation tasks.

These functions can be passed to :class:`isaaclab.managers.ObservationTermCfg`
to specify observations for the policy.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, cast

import torch
import matplotlib.pyplot as plt

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

import engineai_rl_lab.tasks.navigation.mdp as mdp

from .depth_utils.camera_config import CameraConfig, DEFAULT_CAMERA_CONFIG, get_camera_config
from .depth_utils.depth_noise_encoder import DepthNoiseEncoder
from .heightscan_utils.heightscan_encoder import HeightScanFeatEncoder

if TYPE_CHECKING:
    from engineai_rl_lab.tasks.navigation.mdp import PerceptiveNavigationSE2Action
    from engineai_rl_lab.tasks.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

# Visualization state
_DEPTH_VIZ_WINDOWS: dict[str, dict] = {}

# Global depth noise generator - will be initialized with camera config
DEPTH_NOISE_GENERATOR: Optional[DepthNoiseEncoder] = None
JIT_DEPTH_NOISE_GENERATOR: Optional[torch.jit.ScriptModule] = None

# Global height scan encoder - initialized once on first use
HEIGHTSCAN_FEAT_ENCODER: Optional[HeightScanFeatEncoder] = None
JIT_HEIGHTSCAN_FEAT_ENCODER: Optional[torch.jit.ScriptModule] = None

# Global flags/params for depth encoding
use_jit = True
min_depth = 0.0
max_depth = 0.0


def _update_depth_window(camera_name: str, depth_tensor: torch.Tensor, height: int, width: int, title: str) -> None:
    """Update (or lazily create) a persistent matplotlib window for a camera.

    This enables continuous, non-blocking visualization across steps for one or more cameras.
    """
    try:
        # Convert to numpy and handle different tensor shapes
        if depth_tensor.dim() == 4:  # [B, 1, H, W]
            depth_np = depth_tensor[0, 0].detach().cpu().numpy()
        elif depth_tensor.dim() == 2:  # [B, H*W]
            depth_np = depth_tensor[0].detach().cpu().numpy().reshape(height, width)
        else:
            return

        # Lazily create window if missing
        if camera_name not in _DEPTH_VIZ_WINDOWS:
            plt.ion()
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(depth_np, cmap="plasma", aspect="equal")
            cbar = plt.colorbar(im, ax=ax, label="Depth (meters)")
            ax.set_title(title)
            ax.set_xlabel("Width (pixels)")
            ax.set_ylabel("Height (pixels)")
            _DEPTH_VIZ_WINDOWS[camera_name] = {"fig": fig, "ax": ax, "im": im, "cbar": cbar}
        else:
            handle = _DEPTH_VIZ_WINDOWS[camera_name]
            im = handle["im"]
            ax = handle["ax"]
            ax.set_title(title)
            im.set_data(depth_np)
            # Optional: keep consistent color scaling per window
            im.set_clim(vmin=depth_np.min(), vmax=depth_np.max())

        # Lightweight draw
        _DEPTH_VIZ_WINDOWS[camera_name]["fig"].canvas.draw_idle()
        plt.pause(0.001)
    except Exception as e:
        print(f"Error updating depth window for {camera_name}: {e}")


def _ensure_depth_noise_generator_initialized(
    camera_config: Optional[CameraConfig] = None,
    use_jit_precompiled: bool = True,
    feature_dim: int = 64,
):
    """Ensure the depth noise generator is initialized with the correct configuration.

    This is called automatically by observation functions that need depth encoding.
    If not explicitly initialized via initialize_depth_noise_generator(), it will use defaults.

    Args:
        camera_config: The camera configuration to use. If None, uses DEFAULT_CAMERA_CONFIG.
        use_jit_precompiled: Whether to use JIT compilation for faster inference.
        feature_dim: Feature dimension for the encoder output.
    """
    global DEPTH_NOISE_GENERATOR, JIT_DEPTH_NOISE_GENERATOR, use_jit, min_depth, max_depth

    # Only initialize if not already done
    if DEPTH_NOISE_GENERATOR is not None:
        return

    # Use provided camera config or default
    config = camera_config if camera_config is not None else DEFAULT_CAMERA_CONFIG

    # Extract camera parameters
    min_depth = config.min_depth
    max_depth = config.max_depth
    resolution = config.resolution

    print("=" * 80)
    print("Initializing depth noise generator for navigation observation:")
    print(f"  Resolution: {resolution}")
    print(f"  Depth range: [{min_depth}, {max_depth}]")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Encoder path: {config.depth_encoder_path}")
    print(f"  Use JIT: {use_jit_precompiled}")

    # Initialize encoder with the new simplified API
    # The DepthNoiseEncoder now takes camera_config directly
    DEPTH_NOISE_GENERATOR = DepthNoiseEncoder(
        feature_dim=feature_dim,
        camera_config=config,
    ).to(torch.device("cuda"))
    DEPTH_NOISE_GENERATOR.eval()

    # Create JIT version for inference (optional optimization)
    use_jit = use_jit_precompiled
    if use_jit:
        # Try to find a JIT compiled version
        jit_path = config.depth_encoder_path.replace('.pth', '_jit.pt') if config.depth_encoder_path else None
        if jit_path and os.path.exists(jit_path):
            print(f"  Loading precompiled JIT model from: {jit_path}")
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.load(jit_path, map_location="cuda")
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.optimize_for_inference(JIT_DEPTH_NOISE_GENERATOR)
        else:
            print(f"  JIT compilation requested but no precompiled model found.")
            print(f"  Creating JIT model from encoder...")
            example_input = torch.randn(1, 1, resolution[1], resolution[0]).cuda()  # (B, C, H, W)
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.trace(DEPTH_NOISE_GENERATOR, example_input)
            JIT_DEPTH_NOISE_GENERATOR = torch.jit.optimize_for_inference(JIT_DEPTH_NOISE_GENERATOR)
    else:
        JIT_DEPTH_NOISE_GENERATOR = DEPTH_NOISE_GENERATOR

    print("  Depth noise generator initialized successfully")
    print("=" * 80)


def initialize_depth_noise_generator(
    camera_config: Optional[CameraConfig] = None,
    robot_name: Optional[str] = None,
    use_jit_precompiled: bool = True,
    feature_dim: int = 64,
):
    """Initialize the depth noise generator with specific configuration.

    This function should be called in the environment's __post_init__() to set up
    the depth encoder before any observations are computed.

    Args:
        camera_config: The camera configuration to use. If None and robot_name is provided,
                      uses the config for that robot. If both are None, uses DEFAULT_CAMERA_CONFIG.
        robot_name: Name of the robot (e.g., 'b2w', 'aow_d'). If provided and
                   camera_config is None, automatically loads the appropriate camera config.
        use_jit_precompiled: Whether to use JIT compilation for faster inference. Defaults to True.
        feature_dim: Feature dimension for the encoder output. Defaults to 64.

    Examples:
        # Using robot name (recommended for multi-robot training)
        initialize_depth_noise_generator(robot_name="b2w")

        # Using explicit camera config
        initialize_depth_noise_generator(camera_config=ZEDX_CAMERA_CONFIG)
    """
    # If camera_config not provided, try to get it from robot_name
    if camera_config is None and robot_name is not None:
        camera_config = get_camera_config(robot_name, use_default_fallback=False)

    _ensure_depth_noise_generator_initialized(camera_config, use_jit_precompiled, feature_dim)


# ============================================================================
# Observation Functions
# ============================================================================


def generated_commands_reshaped(
    env: ManagerBasedRLEnv, command_name: str, unsqueeze_pos: int = 1, flatten: bool = False
) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    if flatten:
        return env.command_manager.get_command(command_name)
    return env.command_manager.get_command(command_name).unsqueeze(unsqueeze_pos)


def base_lin_vel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute delayed root linear velocity.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        asset_cfg: The name of the asset.

    Returns:
        The delayed linear velocity in the asset's root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    return env.delay_manager.compute_delayed_lin_vel(lin_vel)


def base_ang_vel_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute delayed root angular velocity.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        asset_cfg: The name of the asset.

    Returns:
        The delayed angular velocity in the asset's root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    return env.delay_manager.compute_delayed_ang_vel(ang_vel)


def projected_gravity_delayed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute delayed projected gravity.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        asset_cfg: The name of the asset.

    Returns:
        The delayed projected gravity in the asset's root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b

    return env.delay_manager.compute_delayed_projected_gravity(projected_gravity)


def height_scan_feat(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner_critic"), offset: float = 0.5
) -> torch.Tensor:
    """Height scan feature from a ray caster sensor, encoded using a pre-trained VAE encoder.

    The height scan is reshaped to a 51x51 grid and encoded using a VAE encoder that
    outputs a 64-channel, 7x7 spatial feature map (64*7*7 = 3136 features).

    Args:
        env: The environment object.
        sensor_cfg: The configuration of the height scanner sensor.
        offset: Offset to subtract from the height values. Defaults to 0.5.

    Returns:
        The encoded height scan features of shape (num_envs, 3136).
    """
    global HEIGHTSCAN_FEAT_ENCODER, JIT_HEIGHTSCAN_FEAT_ENCODER

    # Initialize the encoder on first call
    if HEIGHTSCAN_FEAT_ENCODER is None:
        print("Initializing height scan feature encoder...")
        HEIGHTSCAN_FEAT_ENCODER = HeightScanFeatEncoder(feature_dim=64).to(torch.device("cuda"))
        HEIGHTSCAN_FEAT_ENCODER.eval()
        JIT_HEIGHTSCAN_FEAT_ENCODER = torch.jit.script(HEIGHTSCAN_FEAT_ENCODER)
        JIT_HEIGHTSCAN_FEAT_ENCODER = torch.jit.optimize_for_inference(JIT_HEIGHTSCAN_FEAT_ENCODER)

    # Get height scanner data
    height_scanner = env.scene.sensors[sensor_cfg.name]

    # Compute height scan: sensor_height - hit_z - offset
    scan_data = height_scanner.data.pos_w[:, 2].unsqueeze(1) - height_scanner.data.ray_hits_w[..., 2] - offset

    # Clamp the height scan data to the range [-5, 5]
    scan_data = torch.clamp(scan_data, min=-5.0, max=5.0)

    # Expected grid size for the height scanner (51x51 = 2601 points)
    H = W = 51

    # Reshape the height scan data to a 2D grid
    scan_data = scan_data.view(-1, H, W)

    # Encode using the pre-trained encoder
    with torch.no_grad():
        encoded_scan = JIT_HEIGHTSCAN_FEAT_ENCODER(scan_data)

    # Flatten and return: (batch, 64, 7, 7) -> (batch, 3136)
    return encoded_scan.view(env.num_envs, -1)


def generated_actions(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """The generated action from action term in the action manager with the given name.

    Args:
        env: The environment object.
        action_name: The name of the action term.

    Returns:
        The processed actions from the action term.
    """
    return env.action_manager.get_term(action_name).processed_actions


def generated_commands_reshaped_delayed(
    env: ManagerBasedRLEnv,
    command_name: str,
    unsqueeze_pos: int = 1,
    flatten: bool = False,
) -> torch.Tensor:
    """The generated command with delay applied.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        command_name: The name of the command term.
        unsqueeze_pos: Position to unsqueeze the command.
        flatten: Whether to flatten the command.

    Returns:
        The delayed command.
    """
    if flatten:
        command = env.command_manager.get_command(command_name)
        return env.delay_manager.compute_delayed_target_position(command)

    command = env.command_manager.get_command(command_name).unsqueeze(unsqueeze_pos)
    return env.delay_manager.compute_delayed_target_position(command)


def last_low_level_action(
    env: ManagerBasedEnv, action_term: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The last low-level action.

    Args:
        env: The environment object.
        action_term: The name of the action term.

    Returns:
        The last low-level action.
    """
    action_term: PerceptiveNavigationSE2Action = env.action_manager._terms[action_term]
    return action_term.low_level_actions[:, asset_cfg.joint_ids]


@torch.inference_mode()
def depth_image_prefect(env, sensor_cfg):
    """Return the perfect (non-noisy) encoded depth image from the camera.

    Args:
        env: The environment object.
        sensor_cfg: The sensor configuration.

    Returns:
        The encoded depth image features.
    """
    # Ensure encoder is initialized
    _ensure_depth_noise_generator_initialized()

    depth_camera = env.scene.sensors[sensor_cfg.name]

    # Get depth image tensor
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)

    # Handle NaN values by replacing them with 50.0
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)

    # Reshape the tensor to [B, 1, H, W]
    H, W = depth_camera.image_shape
    depth_tensor = depth_tensor.view(-1, 1, H, W)

    assert JIT_DEPTH_NOISE_GENERATOR is not None, (
        "Depth encoder JIT model is not initialized. Call initialize_depth_noise_generator first."
    )
    model = cast(torch.jit.ScriptModule, JIT_DEPTH_NOISE_GENERATOR)
    if use_jit:
        depth_tensor[depth_tensor > max_depth] = 0.0
        encoded_depth_tensor = model(depth_tensor)
    else:
        encoded_depth_tensor, _ = model(depth_tensor)

    return encoded_depth_tensor.view(env.num_envs, -1)


@torch.no_grad()
def depth_image_noisy_delayed(
    env, sensor_cfg, visualize: bool = False
):
    """Return the noisy and delayed encoded depth image from the camera.

    Requires env.delay_manager to exist. Each environment has a random delay
    sampled from [0, max_delay] at episode reset.

    Args:
        env: The environment object (must have delay_manager attribute).
        sensor_cfg: The sensor configuration.
        visualize: Whether to visualize the depth image.

    Returns:
        The delayed encoded depth image features.
    """
    # Ensure encoder is initialized
    _ensure_depth_noise_generator_initialized()

    depth_camera = env.scene.sensors[sensor_cfg.name]

    # Get depth image tensor
    depth_tensor = depth_camera.data.output["distance_to_image_plane"].view(env.num_envs, -1)

    # Handle NaN values by replacing them with 50.0
    depth_tensor = torch.nan_to_num(depth_tensor, nan=50.0, posinf=50.0, neginf=0.0)

    # Reshape the tensor to [B, 1, H, W]
    H, W = depth_camera.image_shape
    depth_tensor = depth_tensor.view(-1, 1, H, W)

    assert JIT_DEPTH_NOISE_GENERATOR is not None, (
        "Depth encoder JIT model is not initialized. Call initialize_depth_noise_generator first."
    )
    model = cast(torch.jit.ScriptModule, JIT_DEPTH_NOISE_GENERATOR)
    if use_jit:
        depth_tensor[depth_tensor > max_depth] = 0.0  # depth larger than depth max is invalid
        depth_tensor[depth_tensor < min_depth] = 0.0  # depth smaller than depth min is invalid
        encoded_depth_tensor = model(depth_tensor)
        noisy_depth_tensor = depth_tensor
    else:
        encoded_depth_tensor, noisy_depth_tensor = model(depth_tensor)

    # Continuous visualization (single-env) with persistent windows
    if visualize and env.num_envs == 1 and not use_jit:
        camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "camera"
        _update_depth_window(camera_name, noisy_depth_tensor, H, W, title=f"Depth (Noisy Delayed) - {camera_name}")

    encoded_depth_tensor_reshaped = encoded_depth_tensor.view(env.num_envs, -1)

    # Apply delay using env's delay manager
    camera_name = sensor_cfg.name if hasattr(sensor_cfg, "name") else "depth"
    return env.delay_manager.compute_delayed_depth(encoded_depth_tensor_reshaped, camera_name)


def _lidar_points_in_yaw_frame(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return normalized LiDAR hit points in the sensor's yaw-aligned frame.

    Missing rays are represented by zero points. Valid points are divided by
    the configured maximum range so the policy receives values near [-1, 1].
    """
    lidar = env.scene.sensors[sensor_cfg.name]
    ray_hits_w = lidar.data.ray_hits_w
    if ray_hits_w is None:
        raise RuntimeError(f"LiDAR sensor '{sensor_cfg.name}' has no ray-hit data.")

    num_envs, num_points, _ = ray_hits_w.shape
    offset_pos = torch.tensor(lidar.cfg.offset.pos, device=ray_hits_w.device, dtype=ray_hits_w.dtype)
    offset_pos = offset_pos.unsqueeze(0).expand(num_envs, -1)
    if lidar.cfg.ray_alignment == "yaw":
        sensor_origin_w = lidar.data.pos_w + math_utils.quat_apply_yaw(lidar.data.quat_w, offset_pos)
    elif lidar.cfg.ray_alignment == "base":
        sensor_origin_w = lidar.data.pos_w + math_utils.quat_apply(lidar.data.quat_w, offset_pos)
    else:
        sensor_origin_w = lidar.data.pos_w + offset_pos

    relative_points_w = ray_hits_w - sensor_origin_w.unsqueeze(1)
    valid = torch.isfinite(relative_points_w).all(dim=-1)
    relative_points_w = torch.nan_to_num(relative_points_w, nan=0.0, posinf=0.0, neginf=0.0)

    sensor_quat_w = lidar.data.quat_w
    if lidar.cfg.ray_alignment == "base":
        offset_quat = torch.tensor(lidar.cfg.offset.rot, device=ray_hits_w.device, dtype=ray_hits_w.dtype)
        sensor_quat_w = math_utils.quat_mul(sensor_quat_w, offset_quat.expand(num_envs, -1))
    heading_quat_w = math_utils.yaw_quat(sensor_quat_w)
    heading_quat_w = heading_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
    points_yaw = math_utils.quat_apply_inverse(
        heading_quat_w,
        relative_points_w.reshape(-1, 3),
    ).reshape(num_envs, num_points, 3)

    max_distance = float(lidar.cfg.max_distance)
    points_yaw = points_yaw / max_distance
    points_yaw = torch.where(valid.unsqueeze(-1), points_yaw, torch.zeros_like(points_yaw))
    return points_yaw.reshape(num_envs, num_points * 3)


@torch.inference_mode()
def lidar_point_cloud(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the current normalized, flattened LiDAR point cloud."""
    return _lidar_points_in_yaw_frame(env, sensor_cfg)


@torch.no_grad()
def lidar_point_cloud_delayed(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the normalized LiDAR point cloud with per-environment delay."""
    point_cloud = _lidar_points_in_yaw_frame(env, sensor_cfg)
    return env.delay_manager.compute_delayed_lidar(point_cloud, sensor_cfg.name)


def in_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Check if the robot is within the goal distance threshold.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        goal_cmd_name: The name of the goal command.

    Returns:
        Boolean tensor indicating whether the robot is within the goal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goal_cmd_generator.pos_command_w[:, :2], dim=1, p=2, keepdim=True)
    return distance_goal < distance_threshold


def time_normalized(env: ManagerBasedRLEnv, command_name: str = "robot_goal") -> torch.Tensor:
    """Time normalized to the maximum episode length.

    Args:
        env: The learning environment.
        command_name: The name of the goal command.

    Returns:
        The normalized time (current step / max steps).
    """
    T_max = env.max_episode_length
    if hasattr(env, "episode_length_buf"):
        t = env.episode_length_buf.unsqueeze(-1)
    else:
        t = torch.tensor([0.0]).repeat(env.num_envs, 1).to(env.device)
    return t / T_max
