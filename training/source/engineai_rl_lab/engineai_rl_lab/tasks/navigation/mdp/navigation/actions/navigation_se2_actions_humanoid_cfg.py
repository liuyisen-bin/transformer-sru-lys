"""Configuration for humanoid hierarchical navigation actions."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .navigation_se2_actions_humanoid import HumanoidNavigationSE2Action


@configclass
class HumanoidNavigationSE2ActionCfg(ActionTermCfg):
    """Humanoid locomotion deployment parameters.

    Defaults mirror the EngineAI native SDK ``rl_walking_example`` interface.
    """

    class_type: type[ActionTerm] = HumanoidNavigationSE2Action

    low_level_position_action: ActionTermCfg = MISSING
    low_level_policy_file: str = MISSING
    active_joint_names: list[str] = MISSING

    low_level_decimation: int = 2
    use_raw_actions: bool = True
    scale: list[float] = [1.0, 1.0, 1.0]
    offset: list[float] = [0.0, 0.0, 0.0]
    policy_scaling: list[float] = [1.0, 0.4, 1.0]
    policy_scaling_negative: list[float] | None = None
    policy_distr_type: str = "gaussian"

    num_observations: int = 72
    num_include_obs_steps: int = 15
    observation_scale: list[float] = [1.0] * 22 + [0.05] * 22 + [1.0] * 28
    command_observation_scale: list[float] = [2.0, 2.0, 1.0]
    observation_clip: float = 100.0
    action_clip: float = 100.0

    enable_low_pass_filter: bool = True
    # Equivalent to the SDK's 0.1 Hz command filter when commands update at 5 Hz.
    low_pass_filter_alpha: float = 0.8819113783

    mnn_backend: str = "CPU"
    mnn_precision: str = "normal"
    mnn_num_threads: int = 1
