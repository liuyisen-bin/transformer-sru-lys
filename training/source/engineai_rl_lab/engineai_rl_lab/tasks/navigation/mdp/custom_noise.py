# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom noise configurations for navigation tasks."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, transform_points
from isaaclab.utils.noise import NoiseCfg

if TYPE_CHECKING:
    pass


@torch.jit.script
def goal_to_xyz(goal: torch.Tensor) -> torch.Tensor:
    """Convert goal in (direction, log distance) format to (x, y, z) coordinates."""
    direction = goal[:, :3]  # Extract directional vector
    distance = torch.exp(goal[:, 3]) - 1  # Convert log distance back to distance
    return direction * distance.unsqueeze(-1)  # Scale direction by distance


@torch.jit.script
def xyz_to_goal(xyz: torch.Tensor) -> torch.Tensor:
    """Convert (x, y, z) coordinates to (direction, log distance) format."""
    distance = torch.norm(xyz, dim=1, keepdim=True) + 1e-6  # Calculate distance
    direction = xyz / distance  # Normalize to get directional vector
    distance = torch.log(1 + distance)  # Convert distance to log distance
    return torch.cat([direction, distance], dim=1)  # Concatenate direction and log distance


@torch.inference_mode()
def delta_transformation_noise(data: torch.Tensor, cfg: "DeltaTransformationNoiseCfg") -> torch.Tensor:
    """Delta transformation noise involving random rotation and translation.

    The output is returned in a new tensor (instead of modifying 'data' in-place).
    """
    # Ensure the data has the shape (..., 3) for 3D coordinates, or (direction, log distance)
    if data.shape[-1] not in (3, 4):
        raise ValueError(
            "Data must have shape (..., 3) for 3D coordinates, or (..., 4) for (direction, log distance)."
        )

    # Determine whether data is in (direction, log distance) format
    unit_vec = data.shape[-1] == 4

    # Convert to (x, y, z) if needed
    if unit_vec:
        coordinate = goal_to_xyz(data)
    else:
        coordinate = data

    # Generate small rotation noise (Rx, Ry, Rz) using uniform distribution
    rotation_noise = torch.empty((data.shape[0], 3), device=data.device).uniform_(-cfg.rotation, cfg.rotation)

    # Convert Euler angles (Rx, Ry, Rz) to quaternions
    quat_noise = quat_from_euler_xyz(rotation_noise[:, 0], rotation_noise[:, 1], rotation_noise[:, 2])

    # Generate small translation noise (Tx, Ty, Tz) using uniform distribution
    translation_noise = torch.empty((data.shape[0], 3), device=data.device).uniform_(-cfg.translation, cfg.translation)

    # Apply random rotation + translation
    transformed_data = transform_points(coordinate.unsqueeze(1), translation_noise, quat_noise).squeeze(1)

    # If input was in goal format, convert back
    if unit_vec:
        transformed_data = xyz_to_goal(transformed_data)

    # Randomly apply the noise based on the probability
    random_mask = torch.rand(data.shape[0], device=data.device) < cfg.noise_prob

    # Use torch.where to select transformed values on the mask
    random_mask_expanded = random_mask.unsqueeze(-1)
    output = torch.where(random_mask_expanded, transformed_data, data)

    if cfg.remove_dist:
        if unit_vec:
            # Zero out the distance component by directly assigning 0
            output[..., 3:] = 0
        else:
            # Normalize the vector by its norm (avoiding division by zero)
            norm = torch.norm(output, dim=-1, keepdim=True) + 1e-6
            output = output / norm

    return output


@configclass
class DeltaTransformationNoiseCfg(NoiseCfg):
    """Configuration for small delta transformation noise involving rotation and translation."""

    func = delta_transformation_noise

    rotation: float = 0.1
    """The maximum rotation angle in radians. Defaults to 0.1."""
    translation: float = 0.5
    """The maximum translation in meters. Defaults to 0.5."""
    noise_prob: float = 0.25
    """The probability of applying the noise. Defaults to 0.25."""
    remove_dist: bool = False
    """Whether to remove the distance from the output. Defaults to False."""
