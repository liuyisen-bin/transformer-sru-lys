# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Camera configuration parameters for different robots and camera types.

This module provides camera-specific parameters for depth noise generation and encoding.
"""

import os
from typing import Optional

from isaaclab.utils import configclass

# Local assets directory for this extension
# Path: depth_utils -> mdp -> navigation -> assets/data
_ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "data")
)


def _get_encoder_path(model_filename: str) -> str:
    """Helper function to construct encoder model path.

    Args:
        model_filename: Name of the encoder model file (e.g., 'vae_pretrain_new.pth')

    Returns:
        Full path to the encoder model file
    """
    return os.path.join(_ASSETS_DIR, "Policies", "depth_encoder", model_filename)


@configclass
class CameraConfig:
    """Configuration class for camera parameters.

    This class contains all camera-specific parameters needed for depth noise generation
    and depth encoder initialization.
    """

    # Camera intrinsic parameters
    focal_length: float = 25.0
    baseline: float = 0.12

    # Depth range parameters
    min_depth: float = 0.25
    max_depth: float = 10.0

    # Camera resolution (width, height)
    resolution: tuple[int, int] = (53, 30)

    # Depth encoder model path
    depth_encoder_path: str = ""

    def __post_init__(self):
        """Post-initialization to set default encoder path if not provided."""
        if not self.depth_encoder_path:
            self.depth_encoder_path = _get_encoder_path("vae_pretrain_fuse.pth")


# Predefined camera configurations
ZEDX_CAMERA_CONFIG = CameraConfig(
    focal_length=25.0,
    baseline=0.12,
    min_depth=0.25,
    max_depth=10.0,
    resolution=(64, 40),
    depth_encoder_path=_get_encoder_path("vae_pretrain_new.pth"),
)
"""Configuration for ZedX camera (used with b2w and aow_d robots)."""

REALSENSE_D435I_CAMERA_CONFIG = CameraConfig(
    # Effective focal length after reducing the 192x120 ray image to 64x40.
    focal_length=33.8,
    baseline=0.05,
    min_depth=0.3,
    max_depth=8.0,
    resolution=(64, 40),
    depth_encoder_path=_get_encoder_path("vae_pretrain_new.pth"),
)
"""Configuration for the Intel RealSense D435i mounted on humanoid robots."""

# Default camera configuration
DEFAULT_CAMERA_CONFIG = ZEDX_CAMERA_CONFIG
"""Default camera configuration (ZedX camera settings)."""

# Robot-to-camera mapping
ROBOT_CAMERA_CONFIGS = {
    "b2w": ZEDX_CAMERA_CONFIG,
    "aow_d": ZEDX_CAMERA_CONFIG,
    "pm01": REALSENSE_D435I_CAMERA_CONFIG,
    "t800": REALSENSE_D435I_CAMERA_CONFIG,
}
"""Dictionary mapping robot names to their camera configurations."""


def get_camera_config(robot_name: str, use_default_fallback: bool = False) -> CameraConfig:
    """Get camera configuration for a specific robot.

    Args:
        robot_name: Name of the robot (e.g., 'b2w', 'aow_d')
        use_default_fallback: If True, return DEFAULT_CAMERA_CONFIG when robot not found
                             instead of raising an error (default: False)

    Returns:
        Camera configuration for the specified robot

    Raises:
        KeyError: If robot_name not found and use_default_fallback is False
    """
    if robot_name in ROBOT_CAMERA_CONFIGS:
        return ROBOT_CAMERA_CONFIGS[robot_name]

    if use_default_fallback:
        return DEFAULT_CAMERA_CONFIG

    available_robots = ", ".join(sorted(ROBOT_CAMERA_CONFIGS.keys()))
    raise KeyError(
        f"Robot '{robot_name}' not found in camera configurations. "
        f"Available robots: {available_robots}"
    )
