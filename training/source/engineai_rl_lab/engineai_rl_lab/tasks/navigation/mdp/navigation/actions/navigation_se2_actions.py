# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Navigation SE2 action term for hierarchical control."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from .navigation_se2_actions_cfg import PerceptiveNavigationSE2ActionCfg


class PerceptiveNavigationSE2Action(ActionTerm):
    """Actions to navigate a robot using hierarchical control with a pre-trained locomotion policy."""

    cfg: PerceptiveNavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: PerceptiveNavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Check if policy file exists
        if not check_file_path(cfg.low_level_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.low_level_policy_file}' does not exist.")
        # Load pre-trained locomotion policy
        file_bytes = read_file(self.cfg.low_level_policy_file)
        self.low_level_policy = torch.jit.load(file_bytes, map_location=self.device)
        self.low_level_policy.eval()

        # prepare joint position actions
        self.low_level_position_action_term: ActionTerm = self.cfg.low_level_position_action.class_type(cfg.low_level_position_action, env)
        self.low_level_velocity_action_term: ActionTerm = self.cfg.low_level_velocity_action.class_type(cfg.low_level_velocity_action, env)

        # prepare buffers
        self._action_dim = 3  # [vx, vy, omega]

        # set up buffers
        self._init_buffers()

        # Low-pass filter state for velocity commands
        self._prev_filtered_velocity_commands = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._low_pass_alpha = self.cfg.low_pass_filter_alpha if hasattr(self.cfg, 'low_pass_filter_alpha') else 0.8
        self._enable_low_pass_filter = self.cfg.enable_low_pass_filter if hasattr(self.cfg, 'enable_low_pass_filter') else True
        # Per-environment per-dimension alpha values (initialized to default, can be randomized per episode)
        # Shape: [num_envs, action_dim] where action_dim = 3 (vx, vy, omega)
        self._per_env_per_dim_low_pass_alpha = torch.full((self.num_envs, self._action_dim), self._low_pass_alpha, device=self.device)


    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def filtered_velocity_commands(self) -> torch.Tensor:
        """Get the current filtered (smoothed) velocity commands."""
        return self._prev_filtered_velocity_commands

    @property
    def low_pass_alpha_values(self) -> torch.Tensor:
        """Get the current per-environment per-dimension low-pass filter alpha values.

        Returns:
            torch.Tensor: Alpha values with shape [num_envs, action_dim] where:
                - action_dim = 3 for [vx, vy, omega]
                - Each environment can have different alpha values for each command dimension
        """
        return self._per_env_per_dim_low_pass_alpha

    @property
    def low_level_actions(self) -> torch.Tensor:
        return torch.cat((self._low_level_position_actions, self._low_level_velocity_actions), dim=1)

    @property
    def low_level_position_actions(self) -> torch.Tensor:
        return self._low_level_position_actions

    @property
    def prev_low_level_position_actions(self) -> torch.Tensor:
        return self._prev_low_level_position_actions

    @property
    def low_level_velocity_actions(self) -> torch.Tensor:
        return self._low_level_velocity_actions

    @property
    def prev_low_level_velocity_actions(self) -> torch.Tensor:
        return self._prev_low_level_velocity_actions

    """
    Operations.
    """

    def apply_low_pass_filter(self, velocity_commands: torch.Tensor) -> torch.Tensor:
        """Apply low-pass filter to velocity commands for smoother locomotion.

        The low-pass filter implements exponential smoothing:
        filtered_cmd(t) = alpha * filtered_cmd(t-1) + (1 - alpha) * new_cmd(t)

        Where alpha is the smoothing factor:
        - alpha = 0.0: No smoothing (pass through)
        - alpha = 1.0: Maximum smoothing (no change)
        - alpha = 0.8: Good balance for locomotion (default)

        This implementation supports per-environment per-dimension alpha values, allowing:
        - Different smoothing for vx, vy, and omega in each environment
        - Independent control over linear and angular velocity response

        Args:
            velocity_commands (torch.Tensor): Raw velocity commands [num_envs, 3] (vx, vy, omega)

        Returns:
            torch.Tensor: Filtered velocity commands with same shape as input
        """
        if not self._enable_low_pass_filter:
            return velocity_commands

        # Use per-environment per-dimension alpha values for filtering
        # Shape: [num_envs, action_dim] - already matches velocity_commands shape
        alpha_values = self._per_env_per_dim_low_pass_alpha

        # Apply exponential smoothing (low-pass filter) with per-environment per-dimension alpha
        filtered_commands = (
            alpha_values * self._prev_filtered_velocity_commands
            + (1.0 - alpha_values) * velocity_commands
        )

        # Update previous filtered commands for next iteration
        self._prev_filtered_velocity_commands.copy_(filtered_commands)

        return filtered_commands

    def process_actions(self, actions):
        """Process low-level navigation actions. This function is called with a frequency of 10Hz.

        Args:
            actions (torch.Tensor): The low-level navigation actions.
        """
        # Store the raw low-level navigation actions
        self._raw_navigation_velocity_actions[:] = actions
        # Apply the affine transformations
        if not self.cfg.use_raw_actions:
            self._processed_navigation_velocity_actions = (
                self._raw_navigation_velocity_actions * self._scale + self._offset
            )
        else:
            self._processed_navigation_velocity_actions[:] = self._raw_navigation_velocity_actions

        if self.cfg.policy_distr_type == "gaussian":
            # scale the actions to the range [-1, 1] for gaussian distribution
            self._processed_navigation_velocity_actions = torch.tanh(self._processed_navigation_velocity_actions)
        elif self.cfg.policy_distr_type == "beta":
            # scale the actions to the range [-1, 1] for beta distribution
            self._processed_navigation_velocity_actions = (self._processed_navigation_velocity_actions - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")

        # compute the current speed of the robot to generate low-level actions based on the current speed
        observations = self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
        base_lin_vel = observations[:, 0:3]
        vel_xyz = base_lin_vel.norm(dim=1, keepdim=True)

        # [vx, vy, omega]
        self._processed_navigation_velocity_actions = (self._processed_navigation_velocity_actions + vel_xyz * self._policy_bias) * self._policy_scaling

        # Apply low-pass filter to smooth velocity commands and add delay effect
        self._processed_navigation_velocity_actions = self.apply_low_pass_filter(self._processed_navigation_velocity_actions)

    @torch.inference_mode()
    def apply_actions(self):
        """Apply low-level actions for the simulator to the physics engine. This functions is called with the
        simulation frequency of 200Hz. Since low-level locomotion runs at 50Hz, we need to decimate the actions."""

        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_position_actions[:] = self._low_level_position_actions.clone()
            self._prev_low_level_velocity_actions[:] = self._low_level_velocity_actions.clone()

            # Get low level actions from low level policy
            actions_phase = self.low_level_policy(
                self._env.observation_manager.compute_group(group_name=self.cfg.observation_group)
            )

            # Process actions and bring them in the right order
            self._low_level_position_actions[:] = actions_phase[:, :self.low_level_position_action_term.action_dim]
            self._low_level_velocity_actions[:] = actions_phase[:, self.low_level_position_action_term.action_dim:]

            # Process low level actions
            self.low_level_position_action_term.process_actions(self._low_level_position_actions)
            self.low_level_velocity_action_term.process_actions(self._low_level_velocity_actions)

        # Apply low level actions
        self.low_level_position_action_term.apply_actions()
        self.low_level_velocity_action_term.apply_actions()
        self._counter += 1

    def reset_low_pass_filter(self, env_ids: torch.Tensor):
        """Reset low-pass filter state for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        self._prev_filtered_velocity_commands[env_ids] = 0.0

    """
    Helper functions
    """

    def _init_buffers(self):
        # Prepare buffers
        self._raw_navigation_velocity_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_navigation_velocity_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._low_level_position_actions = torch.zeros(self.num_envs, self.low_level_position_action_term.action_dim, device=self.device)
        self._low_level_velocity_actions = torch.zeros(self.num_envs, self.low_level_velocity_action_term.action_dim, device=self.device)
        self._prev_low_level_position_actions = torch.zeros_like(self._low_level_position_actions)
        self._prev_low_level_velocity_actions = torch.zeros_like(self._low_level_velocity_actions)
        self._low_level_step_dt = self.cfg.low_level_decimation * self._env.physics_dt
        self._counter = 0
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)
        self._policy_scaling = torch.tensor(self.cfg.policy_scaling, device=self.device).repeat(self.num_envs, 1)
        self._policy_bias = torch.zeros(self.num_envs, self._action_dim, device=self.device)
