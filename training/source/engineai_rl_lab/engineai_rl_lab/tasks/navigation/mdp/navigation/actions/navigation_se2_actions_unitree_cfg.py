"""Configuration for Unitree's official G1 velocity policy backend."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .navigation_se2_actions_unitree import UnitreeNavigationSE2Action


@configclass
class UnitreeNavigationSE2ActionCfg(ActionTermCfg):
    """Three-dimensional navigation command with a Unitree 29-DOF backend."""

    class_type: type[ActionTerm] = UnitreeNavigationSE2Action

    low_level_position_action: ActionTermCfg = MISSING
    policy_file: str = MISSING
    joint_names: list[str] = MISSING

    low_level_decimation: int = 4
    # Match the official keyboard Q/E amplitude, rather than its unused YAML yaw limit.
    command_min: list[float] = [-0.5, -0.3, -1.0]
    command_max: list[float] = [1.0, 0.3, 1.0]
    policy_scaling: list[float] = [1.0, 1.0, 1.0]
    enable_low_pass_filter: bool = True
    # [vx, vy, wz]: preserve translation smoothing; yaw follows each 5 Hz command directly.
    low_pass_filter_alpha: list[float] = [0.8819113783, 0.8819113783, 0.0]


__all__ = ["UnitreeNavigationSE2ActionCfg"]
