# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom math utilities for navigation tasks."""
from __future__ import annotations

import torch


@torch.jit.script
def vec_to_quat(v: torch.Tensor) -> torch.Tensor:
    """Converts a unit vector to a quaternion representing the rotation from the X-axis to the vector v.

    Args:
        v (torch.Tensor): A tensor of shape (..., 3) representing the unit vectors.

    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing the quaternions (w, x, y, z).
    """
    # Ensure v is a unit vector
    norm_v = v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)

    # Reference vector (X-axis)
    a = torch.tensor([1.0, 0.0, 0.0], device=norm_v.device, dtype=norm_v.dtype).expand_as(norm_v)

    # Compute dot and cross products
    dot = torch.sum(a * norm_v, dim=-1, keepdim=True)
    w = torch.cross(a, norm_v, dim=-1)

    # Compute quaternion components
    s = torch.sqrt((1.0 + dot) * 0.5).clamp(min=1e-8)  # Avoid division by zero
    q = torch.cat([s, w / (2.0 * s)], dim=-1)

    # Handle edge case when dot == -1 (vectors are opposite)
    mask = (dot < -0.999999).squeeze(-1)
    if mask.any():
        # Choose an arbitrary orthogonal vector
        orthogonal = torch.cross(
            a[mask],
            torch.tensor([0.0, 1.0, 0.0], device=norm_v.device, dtype=norm_v.dtype).expand_as(a[mask]),
            dim=-1
        )
        orthogonal = orthogonal / torch.clamp(torch.norm(orthogonal, dim=-1, keepdim=True), min=1e-8)
        q[mask] = torch.cat([torch.zeros_like(s[mask]), orthogonal], dim=-1)

    # Normalize the quaternion
    q = q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)
    return q
