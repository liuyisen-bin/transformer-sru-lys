# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_inv, yaw_quat, quat_mul, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from engineai_rl_lab.tasks.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand


def euler_xyz_from_quat_wrapped(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion to Euler angles (XYZ convention) with wrapping to [-pi, pi].

    Args:
        quat: Quaternion tensor of shape (..., 4) in (w, x, y, z) format.

    Returns:
        Tuple of (roll, pitch, yaw) tensors.
    """
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    # Wrap to [-pi, pi]
    roll = torch.remainder(roll + torch.pi, 2 * torch.pi) - torch.pi
    pitch = torch.remainder(pitch + torch.pi, 2 * torch.pi) - torch.pi
    yaw = torch.remainder(yaw + torch.pi, 2 * torch.pi) - torch.pi
    return roll, pitch, yaw


def time_out_navigation(
    env: "ManagerBasedRLEnv",
    goal_cmd_name: str = "robot_goal",
    distance_threshold: float = 0.5
) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length.

    This also tracks success metrics by checking if the robot reached the goal before timeout.
    """
    from engineai_rl_lab.tasks.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]

    termination = env.episode_length_buf >= env.max_episode_length

    env_ids = torch.where(termination)[0]

    distance_goal = torch.norm(goal_cmd_generator._get_unscaled_command()[:, :2], dim=1)

    # Track continuous time inside the goal region. Leaving the region resets
    # the dwell timer instead of permanently latching the goal as reached.
    at_goal = distance_goal < distance_threshold
    goal_cmd_generator.time_at_goal[~at_goal] = 0.0
    goal_cmd_generator.time_at_goal[at_goal] += env.step_dt

    if env_ids.numel() > 0:  # Check if env_ids is not empty
        required_goal_time = goal_cmd_generator.required_time_at_goal_in_steps * env.step_dt
        success_masks = goal_cmd_generator.time_at_goal >= required_goal_time
        value_buffer = torch.zeros_like(distance_goal)  # init with 0: Fail
        value_buffer[success_masks] = 1.0  # Success
        goal_cmd_generator.goal_reached_buffer.add(value_buffer, env_ids)

    return termination


def illegal_contact_navigation(
    env: "ManagerBasedRLEnv",
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    from engineai_rl_lab.tasks.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]

    termination = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )

    env_ids = torch.where(termination)[0]

    if env_ids.numel() > 0:  # Check if env_ids is not empty
        goal_cmd_generator.goal_reached_buffer.add(torch.zeros_like(termination, dtype=torch.float), env_ids)

    return termination


def large_angle_termination_navigation(
    env: "ManagerBasedRLEnv",
    threshold: float,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate when the robot exceeds a pitch or roll angle threshold."""
    from engineai_rl_lab.tasks.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms[goal_cmd_name]

    # degree to rad
    threshold_rad = threshold * torch.pi / 180.0

    robot = env.scene["robot"]
    yaw_q = yaw_quat(robot.data.root_quat_w)
    base_quat_b = quat_mul(quat_inv(yaw_q), robot.data.root_quat_w)
    robot_roll, robot_pitch, _ = euler_xyz_from_quat_wrapped(base_quat_b)

    termination = torch.logical_or(torch.abs(robot_pitch) > threshold_rad, torch.abs(robot_roll) > threshold_rad)

    env_ids = torch.where(termination)[0]

    if env_ids.numel() > 0:  # Check if env_ids is not empty
        goal_cmd_generator.goal_reached_buffer.add(torch.zeros_like(termination, dtype=torch.float), env_ids)

    return termination


def at_goal_navigation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.5,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate the episode when the goal is reached.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.
        goal_cmd_name: The name of the goal command term.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    from engineai_rl_lab.tasks.navigation.mdp.navigation.goal_commands import RobotNavigationGoalCommand

    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    goal_cmd_generator: RobotNavigationGoalCommand = env.command_manager._terms.get(goal_cmd_name)

    # Calculate distance to goal
    xy_error = torch.norm(goal_cmd_generator._get_unscaled_command()[:, :2], dim=1)

    # A successful episode requires an uninterrupted dwell inside the goal
    # region. Reset progress immediately if the robot leaves the region.
    at_goal = xy_error < distance_threshold
    dwell_steps = goal_cmd_generator.time_at_goal_in_steps
    dwell_steps[~at_goal] = 0
    dwell_steps[at_goal] += 1

    # Determine if termination condition is met
    termination = dwell_steps >= goal_cmd_generator.required_time_at_goal_in_steps

    # Update goal reached buffer if termination condition is met
    env_ids = torch.where(termination)[0]
    if env_ids.numel() > 0:  # Check if any environments have met the termination condition
        goal_cmd_generator.goal_reached_buffer.add(torch.ones_like(termination, dtype=torch.float), env_ids)

    return termination


def terrain_fall(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fall_height_threshold: float = -1.0,
    goal_cmd_name: str = "robot_goal",
) -> torch.Tensor:
    """Terminate when the robot falls below a certain height threshold.

    This termination is triggered when the robot's z-position falls below a
    specified threshold, indicating that the robot has fallen off the terrain
    or into a deep pit.

    Args:
        env: The learning environment.
        asset_cfg: The configuration for the robot asset.
        fall_height_threshold: The z-height below which the robot is considered fallen (in meters).
            Default is -1.0m to account for pits which can be ~1.5m deep.
        goal_cmd_name: The name of the goal command term.

    Returns:
        Boolean tensor indicating whether the robot has fallen.
    """
    # Direct tensor access for z-coordinate (avoids intermediate variable allocation)
    termination = env.scene[asset_cfg.name].data.root_pos_w[:, 2] < fall_height_threshold

    # Early exit if no terminations (common case - avoids torch.where overhead)
    if not termination.any():
        return termination

    # Update goal reached buffer with failure
    goal_cmd = env.command_manager._terms.get(goal_cmd_name)
    if goal_cmd is not None:
        env_ids = termination.nonzero(as_tuple=False).squeeze(-1)
        goal_cmd.goal_reached_buffer.add(torch.zeros(env.num_envs, dtype=torch.float, device=env.device), env_ids)

    return termination
