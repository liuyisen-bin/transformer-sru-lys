# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Standalone observation delay manager for simulating sensor delays."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from isaaclab.utils import DelayBuffer, configclass


@configclass
class ObservationDelayManagerCfg:
    """Configuration for the observation delay manager.

    This configuration controls the simulation of sensor delays, where each environment
    can have a different random delay sampled from [0, max_delay] at episode reset.
    """

    enabled: bool = True
    """Whether to enable delay simulation for observations."""

    max_delay_lin_vel: int = 2
    """Maximum delay in timesteps for linear velocity observations."""

    max_delay_ang_vel: int = 2
    """Maximum delay in timesteps for angular velocity observations."""

    max_delay_projected_gravity: int = 2
    """Maximum delay in timesteps for projected gravity observations."""

    max_delay_target_position: int = 2
    """Maximum delay in timesteps for target position/goal observations."""

    max_delay_depth: int = 2
    """Maximum delay in timesteps for depth image observations."""

    max_delay_lidar: int = 2
    """Maximum delay in timesteps for LiDAR point-cloud observations."""


@dataclass
class DelayBufferState:
    """Container for a delay buffer and its associated time lags.

    This encapsulates the state needed for delayed observations:
    - The buffer storing historical data
    - Per-environment time lags (delays)
    - Maximum delay configured for this buffer
    """

    buffer: DelayBuffer
    time_lags: torch.Tensor
    max_delay: int

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append data and return delayed version."""
        return self.buffer.compute(data)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset buffer for specified environments."""
        self.buffer.reset(env_ids)

    def randomize_lags(self, env_ids: torch.Tensor, device: str):
        """Randomize time lags for specified environments."""
        new_lags = torch.randint(0, self.max_delay + 1, (len(env_ids),), dtype=torch.int, device=device)
        self.time_lags[env_ids] = new_lags
        env_ids_list = env_ids.tolist() if env_ids is not None else None
        self.buffer.set_time_lag(new_lags, env_ids_list)

    def set_lags(self, time_lags: torch.Tensor, env_ids: torch.Tensor | None = None):
        """Set time lags for specified environments."""
        if env_ids is None:
            self.time_lags[:] = time_lags
            self.buffer.set_time_lag(time_lags)
        else:
            self.time_lags[env_ids] = time_lags
            self.buffer.set_time_lag(time_lags, env_ids.tolist())


class ObservationDelayManager:
    """Standalone manager for simulating observation delays.

    This manager can be added to any environment to simulate sensor delays
    without coupling to a specific action term. Each observation type can
    have its own delay buffer with per-environment random delays.

    Usage:
        # In environment config or __init__:
        self.delay_manager = ObservationDelayManager(cfg, num_envs, device)

        # In observation functions:
        delayed_vel = env.delay_manager.compute_delayed_lin_vel(current_vel)

        # On episode reset:
        env.delay_manager.reset(env_ids)
        env.delay_manager.randomize_lags(env_ids)
    """

    def __init__(self, cfg: ObservationDelayManagerCfg, num_envs: int, device: str):
        """Initialize the observation delay manager.

        Args:
            cfg: Configuration for delay buffers.
            num_envs: Number of parallel environments.
            device: Device for tensor operations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self._delay_enabled = cfg.enabled

        self._init_delay_buffers()

    def _create_delay_buffer_state(self, max_delay: int) -> DelayBufferState:
        """Create a DelayBufferState with initialized buffer and random time lags."""
        buffer = DelayBuffer(max_delay, batch_size=self.num_envs, device=self.device)
        time_lags = torch.randint(0, max_delay + 1, (self.num_envs,), dtype=torch.int, device=self.device)
        buffer.set_time_lag(time_lags)
        return DelayBufferState(buffer=buffer, time_lags=time_lags, max_delay=max_delay)

    def _init_delay_buffers(self):
        """Initialize all delay buffers based on configuration."""
        if not self.cfg.enabled:
            # No delays - set all delay states to None
            self._delay_lin_vel: DelayBufferState | None = None
            self._delay_ang_vel: DelayBufferState | None = None
            self._delay_projected_gravity: DelayBufferState | None = None
            self._delay_target_position: DelayBufferState | None = None
            self._delay_depth_buffers: dict[str, DelayBufferState] = {}
            self._delay_lidar_buffers: dict[str, DelayBufferState] = {}
            self._max_delay_depth = 0
            self._max_delay_lidar = 0
            return

        # Initialize delay buffer states for each observation type
        self._delay_lin_vel = self._create_delay_buffer_state(self.cfg.max_delay_lin_vel)
        self._delay_ang_vel = self._create_delay_buffer_state(self.cfg.max_delay_ang_vel)
        self._delay_projected_gravity = self._create_delay_buffer_state(self.cfg.max_delay_projected_gravity)
        self._delay_target_position = self._create_delay_buffer_state(self.cfg.max_delay_target_position)
        # Depth buffers are created dynamically per camera
        self._delay_depth_buffers: dict[str, DelayBufferState] = {}
        self._max_delay_depth = self.cfg.max_delay_depth
        # LiDAR buffers are also created lazily, one per sensor.
        self._delay_lidar_buffers: dict[str, DelayBufferState] = {}
        self._max_delay_lidar = self.cfg.max_delay_lidar

    @property
    def enabled(self) -> bool:
        """Whether delay simulation is enabled."""
        return self._delay_enabled

    def get_or_create_depth_delay_buffer(self, camera_name: str) -> DelayBufferState:
        """Get or create a delay buffer for a specific depth camera.

        Args:
            camera_name: Name of the depth camera sensor.

        Returns:
            DelayBufferState for the specified camera.
        """
        if camera_name not in self._delay_depth_buffers:
            self._delay_depth_buffers[camera_name] = self._create_delay_buffer_state(self._max_delay_depth)
        return self._delay_depth_buffers[camera_name]

    def get_or_create_lidar_delay_buffer(self, sensor_name: str) -> DelayBufferState:
        """Get or create a delay buffer for a LiDAR sensor."""
        if sensor_name not in self._delay_lidar_buffers:
            self._delay_lidar_buffers[sensor_name] = self._create_delay_buffer_state(self._max_delay_lidar)
        return self._delay_lidar_buffers[sensor_name]

    def reset(self, env_ids: torch.Tensor):
        """Reset all delay buffers for specified environments.

        Args:
            env_ids: Environment indices to reset.
        """
        env_ids_list = env_ids.tolist() if env_ids is not None else None

        if self._delay_lin_vel is not None:
            self._delay_lin_vel.reset(env_ids_list)
        if self._delay_ang_vel is not None:
            self._delay_ang_vel.reset(env_ids_list)
        if self._delay_projected_gravity is not None:
            self._delay_projected_gravity.reset(env_ids_list)
        if self._delay_target_position is not None:
            self._delay_target_position.reset(env_ids_list)

        # Reset all depth buffers
        for depth_buffer in self._delay_depth_buffers.values():
            depth_buffer.reset(env_ids_list)
        for lidar_buffer in self._delay_lidar_buffers.values():
            lidar_buffer.reset(env_ids_list)

    def randomize_lags(self, env_ids: torch.Tensor):
        """Randomize time lags for all delay buffers.

        Args:
            env_ids: Environment indices to randomize.
        """
        if self._delay_lin_vel is not None:
            self._delay_lin_vel.randomize_lags(env_ids, self.device)
        if self._delay_ang_vel is not None:
            self._delay_ang_vel.randomize_lags(env_ids, self.device)
        if self._delay_projected_gravity is not None:
            self._delay_projected_gravity.randomize_lags(env_ids, self.device)
        if self._delay_target_position is not None:
            self._delay_target_position.randomize_lags(env_ids, self.device)

        # Randomize all depth buffers
        for depth_buffer in self._delay_depth_buffers.values():
            depth_buffer.randomize_lags(env_ids, self.device)
        for lidar_buffer in self._delay_lidar_buffers.values():
            lidar_buffer.randomize_lags(env_ids, self.device)

    # ============================================================================
    # Delayed observation computation methods
    # ============================================================================

    def compute_delayed_lin_vel(self, lin_vel: torch.Tensor) -> torch.Tensor:
        """Compute delayed linear velocity observation.

        Args:
            lin_vel: Current linear velocity tensor of shape (num_envs, 3).

        Returns:
            Delayed linear velocity, or original if delays disabled.
        """
        if self._delay_lin_vel is None:
            return lin_vel
        return self._delay_lin_vel.compute(lin_vel)

    def compute_delayed_ang_vel(self, ang_vel: torch.Tensor) -> torch.Tensor:
        """Compute delayed angular velocity observation.

        Args:
            ang_vel: Current angular velocity tensor of shape (num_envs, 3).

        Returns:
            Delayed angular velocity, or original if delays disabled.
        """
        if self._delay_ang_vel is None:
            return ang_vel
        return self._delay_ang_vel.compute(ang_vel)

    def compute_delayed_projected_gravity(self, projected_gravity: torch.Tensor) -> torch.Tensor:
        """Compute delayed projected gravity observation.

        Args:
            projected_gravity: Current projected gravity tensor of shape (num_envs, 3).

        Returns:
            Delayed projected gravity, or original if delays disabled.
        """
        if self._delay_projected_gravity is None:
            return projected_gravity
        return self._delay_projected_gravity.compute(projected_gravity)

    def compute_delayed_target_position(self, target_position: torch.Tensor) -> torch.Tensor:
        """Compute delayed target position observation.

        Args:
            target_position: Current target position tensor.

        Returns:
            Delayed target position, or original if delays disabled.
        """
        if self._delay_target_position is None:
            return target_position
        return self._delay_target_position.compute(target_position)

    def compute_delayed_depth(self, depth_features: torch.Tensor, camera_name: str) -> torch.Tensor:
        """Compute delayed depth observation for a specific camera.

        Args:
            depth_features: Encoded depth features of shape (num_envs, feature_dim).
            camera_name: Name of the camera sensor.

        Returns:
            Delayed depth features, or original if delays disabled.
        """
        if not self._delay_enabled:
            return depth_features
        delay_buffer = self.get_or_create_depth_delay_buffer(camera_name)
        return delay_buffer.compute(depth_features)

    def compute_delayed_lidar(self, point_cloud: torch.Tensor, sensor_name: str) -> torch.Tensor:
        """Compute a delayed flattened point cloud for one LiDAR sensor."""
        if not self._delay_enabled:
            return point_cloud
        delay_buffer = self.get_or_create_lidar_delay_buffer(sensor_name)
        return delay_buffer.compute(point_cloud)
