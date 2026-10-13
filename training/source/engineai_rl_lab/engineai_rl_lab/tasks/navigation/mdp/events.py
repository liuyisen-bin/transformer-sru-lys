# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom event functions for navigation tasks.

These functions are ported from the internal Isaac Lab fork and provide
domain randomization and environment events specific to navigation tasks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.sensors import RayCasterCamera
    from engineai_rl_lab.tasks.navigation.mdp import PerceptiveNavigationSE2Action


def randomize_camera_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    height_range: float,
    roll_angle_range: float,
    pitch_angle_range: float,
    yaw_angle_range: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("raycast_camera"),
):
    """Randomize the camera's height and orientation around the original values.

    The function samples around the original camera offset and orientation:
    - height: original height +- height_range (meters)
    - roll: original roll +- roll_angle_range (degrees)
    - pitch: original pitch +- pitch_angle_range (degrees)
    - yaw: original yaw +- yaw_angle_range (degrees)

    Args:
        env: The environment manager.
        env_ids: The environment ids to randomize.
        height_range: Range around original height (meters).
        roll_angle_range: Range around original roll angle (degrees).
        pitch_angle_range: Range around original pitch angle (degrees).
        yaw_angle_range: Range around original yaw angle (degrees).
        asset_cfg: Scene entity configuration for the camera.
    """
    camera: RayCasterCamera = env.scene[asset_cfg.name]

    # Get original camera position and orientation
    original_camera_pos = camera.cfg.offset.pos
    original_camera_quat = camera.cfg.offset.rot

    # Sample position around original value
    num_envs = env_ids.numel()
    pos_delta = torch.zeros((num_envs, 3), device=env.device)
    pos_delta[:, 0] = torch.randn(num_envs, device=env.device) * (height_range / 2.0)
    pos_delta[:, 1] = torch.randn(num_envs, device=env.device) * (height_range / 2.0)
    pos_delta[:, 0] = torch.clamp(pos_delta[:, 0], -height_range, height_range)
    pos_delta[:, 1] = torch.clamp(pos_delta[:, 1], -height_range, height_range)
    pos_delta[:, 2] = torch.rand(num_envs, device=env.device) * 2.0 * height_range - height_range
    camera._offset_pos[env_ids] = torch.tensor(original_camera_pos, device=env.device) + pos_delta

    # Convert original quaternion to Euler angles (XYZ convention)
    quat_tensor = torch.tensor(original_camera_quat, device=env.device).unsqueeze(0)
    original_euler = math_utils.euler_xyz_from_quat(quat_tensor)
    original_roll_deg = original_euler[0] * 180.0 / torch.pi
    original_pitch_deg = original_euler[1] * 180.0 / torch.pi
    original_yaw_deg = original_euler[2] * 180.0 / torch.pi

    # Sample angles around original values
    roll_delta_deg = torch.randn(num_envs, device=env.device) * (roll_angle_range / 2.0)
    roll_delta_deg = torch.clamp(roll_delta_deg, -roll_angle_range, roll_angle_range)
    pitch_delta_deg = torch.rand(num_envs, device=env.device) * 2.0 * pitch_angle_range - pitch_angle_range
    yaw_delta_deg = torch.randn(num_envs, device=env.device) * (yaw_angle_range / 2.0)
    yaw_delta_deg = torch.clamp(yaw_delta_deg, -yaw_angle_range, yaw_angle_range)

    # Add deltas to original angles
    sampled_roll_deg = original_roll_deg + roll_delta_deg
    sampled_pitch_deg = original_pitch_deg + pitch_delta_deg
    sampled_yaw_deg = original_yaw_deg + yaw_delta_deg

    # Convert to radians and build quaternion (XYZ convention)
    sampled_roll_rad = sampled_roll_deg * torch.pi / 180.0
    sampled_pitch_rad = sampled_pitch_deg * torch.pi / 180.0
    sampled_yaw_rad = sampled_yaw_deg * torch.pi / 180.0

    camera._offset_quat[env_ids] = math_utils.quat_from_euler_xyz(sampled_roll_rad, sampled_pitch_rad, sampled_yaw_rad)


def randomize_action_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    scale_range_x: tuple[float, float],
    scale_range_y: tuple[float, float],
    scale_range_theta: tuple[float, float],
    scale_range_xb: float,
    scale_range_yb: float,
    scale_range_thetab: float,
    action_term: str,
):
    """Change the action scale in the scene.

    Args:
        env: The environment manager.
        env_ids: The environment ids to change the action scale in.
        scale_range_x: The scale range for x.
        scale_range_y: The scale range for y.
        scale_range_theta: The scale range for theta.
        scale_range_xb: The bias range for x.
        scale_range_yb: The bias range for y.
        scale_range_thetab: The bias range for theta.
        action_term: Name of the action term.
    """
    action_term_obj: PerceptiveNavigationSE2Action = env.action_manager._terms[action_term]

    action_term_obj._policy_scaling[env_ids] = torch.tensor(
        action_term_obj.cfg.policy_scaling, device=action_term_obj.device
    ).expand(len(env_ids), -1)

    random_scales_x = (
        torch.rand(len(env_ids), 1, device=env.device) * (scale_range_x[1] - scale_range_x[0]) + scale_range_x[0]
    )
    random_scales_y = (
        torch.rand(len(env_ids), 1, device=env.device) * (scale_range_y[1] - scale_range_y[0]) + scale_range_y[0]
    )
    random_scales_theta = (
        torch.rand(len(env_ids), 1, device=env.device) * (scale_range_theta[1] - scale_range_theta[0])
        + scale_range_theta[0]
    )
    random_scales_xb = (torch.rand(len(env_ids), 1, device=env.device) * 2 - 1) * scale_range_xb
    random_scales_yb = (torch.rand(len(env_ids), 1, device=env.device) * 2 - 1) * scale_range_yb
    random_scales_thetab = (torch.rand(len(env_ids), 1, device=env.device) * 2 - 1) * scale_range_thetab

    random_scales = torch.cat((random_scales_x, random_scales_y, random_scales_theta), dim=-1)
    random_scales_b = torch.cat((random_scales_xb, random_scales_yb, random_scales_thetab), dim=-1)

    action_term_obj._policy_scaling[env_ids] *= random_scales
    action_term_obj._policy_bias[env_ids, :] = random_scales_b


def reset_and_randomize_delay_buffer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """Reset and randomize delay buffers.

    This function resets the delay buffers and randomizes the time lags for
    all observation types. Requires env.delay_manager to exist.

    Args:
        env: The environment manager (must have delay_manager attribute).
        env_ids: The environment ids to reset.
    """
    env.delay_manager.reset(env_ids)
    env.delay_manager.randomize_lags(env_ids)


def randomize_low_pass_filter_alpha(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    alpha_range: tuple[float, float],
    action_term: str,
    per_dimension: bool = True,
    alpha_range_vx: tuple[float, float] | None = None,
    alpha_range_vy: tuple[float, float] | None = None,
    alpha_range_omega: tuple[float, float] | None = None,
):
    """Randomize the low-pass filter alpha parameter for velocity command smoothing.

    Args:
        env: The environment manager.
        env_ids: The environment ids to randomize the alpha parameter for.
        alpha_range: Tuple of (min_alpha, max_alpha) values.
        action_term: Name of the action term to modify.
        per_dimension: Whether to use different alpha for each dimension.
        alpha_range_vx: Optional range for vx dimension.
        alpha_range_vy: Optional range for vy dimension.
        alpha_range_omega: Optional range for omega dimension.
    """
    action_term_obj: PerceptiveNavigationSE2Action = env.action_manager._terms[action_term]

    if per_dimension:
        # Use dimension-specific ranges or fall back to global range
        vx_range = alpha_range_vx if alpha_range_vx is not None else alpha_range
        vy_range = alpha_range_vy if alpha_range_vy is not None else alpha_range
        omega_range = alpha_range_omega if alpha_range_omega is not None else alpha_range

        alpha_vx = torch.rand(len(env_ids), device=env.device) * (vx_range[1] - vx_range[0]) + vx_range[0]
        alpha_vy = torch.rand(len(env_ids), device=env.device) * (vy_range[1] - vy_range[0]) + vy_range[0]
        alpha_omega = torch.rand(len(env_ids), device=env.device) * (omega_range[1] - omega_range[0]) + omega_range[0]

        if hasattr(action_term_obj, "_per_env_per_dim_low_pass_alpha"):
            action_term_obj._per_env_per_dim_low_pass_alpha[env_ids, 0] = alpha_vx
            action_term_obj._per_env_per_dim_low_pass_alpha[env_ids, 1] = alpha_vy
            action_term_obj._per_env_per_dim_low_pass_alpha[env_ids, 2] = alpha_omega
    else:
        random_alpha = (
            torch.rand(len(env_ids), device=env.device) * (alpha_range[1] - alpha_range[0]) + alpha_range[0]
        )
        if hasattr(action_term_obj, "_per_env_per_dim_low_pass_alpha"):
            action_term_obj._per_env_per_dim_low_pass_alpha[env_ids] = random_alpha.unsqueeze(-1).expand(-1, 3)


def disable_backward_penalty_after_steps(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    disable_after_steps: int,
    action_term: str,
):
    """Disable backward movement penalty after a certain number of steps.

    Args:
        env: The environment manager.
        env_ids: The environment ids.
        disable_after_steps: Number of steps after which to disable the penalty.
        action_term: Name of the action term.
    """
    action_term_obj: PerceptiveNavigationSE2Action = env.action_manager._terms[action_term]

    if hasattr(action_term_obj, "disable_backward_penalty"):
        # Check which environments have exceeded the step threshold
        if hasattr(env, "episode_length_buf"):
            exceeded_steps = env.episode_length_buf[env_ids] >= disable_after_steps
            action_term_obj.disable_backward_penalty[env_ids[exceeded_steps]] = True
