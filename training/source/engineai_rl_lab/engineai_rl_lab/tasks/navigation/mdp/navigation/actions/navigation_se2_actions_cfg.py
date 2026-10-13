# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .navigation_se2_actions import PerceptiveNavigationSE2Action


@configclass
class PerceptiveNavigationSE2ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PerceptiveNavigationSE2Action
    """ Class of the action term."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    use_raw_actions: bool = False
    """Whether to use raw actions or not."""
    scale: list[float] = [1.0, 1.0, 1.0]
    """Scale for the actions [vx, vy, w]."""
    offset: list[float] = [0.0, 0.0, 0.0]
    """Offset for the actions [vx, vy, w]."""
    low_level_velocity_action: ActionTermCfg = MISSING
    """Configuration of the low level velocity action term."""
    low_level_position_action: ActionTermCfg = MISSING
    """Configuration of the low level position action term."""
    low_level_policy_file: str = MISSING
    """Path to the low level policy file."""
    observation_group: str = "policy"
    """Observation group to use for the low level policy."""
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    """Policy dependent scaling for the actions [vx, vy, w]."""
    reorder_joint_list: list[str] | None = None
    """Reorder the joint actions given from the low-level policy to match the Isaac Sim order if policy has been
    trained with a different order. Set to None to disable reordering."""
    policy_distr_type: str = "gaussian"
    """Policy distribution type: 'gaussian', 'beta'."""
    # Low-pass filter parameters
    enable_low_pass_filter: bool = True
    """Whether to enable low-pass filtering for velocity commands."""
    low_pass_filter_alpha: float = 0.5
    """Low-pass filter smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing).
    Formula: filtered_cmd = alpha * prev_filtered_cmd + (1 - alpha) * new_cmd"""
